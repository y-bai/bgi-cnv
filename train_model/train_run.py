#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: train_run.py
    Description:
    
Created by Yong Bai on 2019/8/20 2:38 PM.
"""
import os
import numpy as np
import pandas as pd
import shutil
from model import cnv_net, cnv_aenet
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger
from model_utils import AdvancedLearnignRateScheduler, MultiGPUCheckpointCallback, CyclicLR
from keras.utils import multi_gpu_model
from keras import backend as K
from train_model import CNVDataGenerator, CNVDataGenerator2, CNVDataGenerator3, CNVDataGenerator_AE
from itertools import product
from functools import partial
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py


# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
# https://github.com/keras-team/keras/issues/9395#issuecomment-379228094
# https://blog.csdn.net/m0_37477175/article/details/83004746
# https://www.groundai.com/project/
# a-novel-focal-tversky-loss-function-with-improved-attention-u-net-for-lesion-segmentation/1
def tversky_loss(alpha=0.5, beta=0.5):
    alpha = float(alpha)  # weight to FP
    beta = float(beta)  # weight to FN

    def tversky_loss_fixed(y_true, y_pred):
        ones = K.ones(K.shape(y_true))
        p0 = y_pred  # proba that are class i
        p1 = ones - y_pred  # proba that are not class i
        g0 = y_true
        g1 = ones - y_true

        num = K.sum(p0 * g0, axis=0)
        den = num + alpha * K.sum(p0 * g1, axis=0) + beta * K.sum(p1 * g0, axis=0)
        # num = K.sum(p0 * g0, (0, 1))
        # den = num + alpha * K.sum(p0 * g1, (0, 1)) + beta * K.sum(p1 * g0, (0, 1))

        T = K.sum((num + 1e-6) / (den + 1e-6))  # when summing over classes, T has dynamic range [0 Ncl]

        Ncl = K.cast(K.shape(y_true)[-1], 'float32')
        return Ncl - T

    return tversky_loss_fixed


def focal_loss(gamma=2., alpha=4.):
    gamma = float(gamma)
    alpha = float(alpha)

    # https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
    #

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)

    return focal_loss_fixed


def combine_tversky_focal_loss(alpha_t=0.5, beta_t=0.5, gamma_f=2., alpha_f=4.):
    t_loss = tversky_loss(alpha=alpha_t, beta=beta_t)
    f_loss = focal_loss(gamma=gamma_f, alpha=alpha_f)

    def combine_tversky_focal_loss_fixed(y_true, y_pred):
        return t_loss(y_true, y_pred) + f_loss(y_true, y_pred)

    return combine_tversky_focal_loss_fixed


def combine_weighted_ce_focal_loss(weights=None, gamma_f=2., alpha_f=4.):
    if weights is None:
        weighted_ce = K.categorical_crossentropy
    else:
        weighted_ce = partial(w_categorical_crossentropy, weights=weights)
    f_loss = focal_loss(gamma=gamma_f, alpha=alpha_f)

    def combine_weighted_ce_focal_loss_fixed(y_true, y_pred):
        return weighted_ce(y_true, y_pred) + f_loss(y_true, y_pred)

    return combine_weighted_ce_focal_loss_fixed


def w_categorical_crossentropy(y_true, y_pred, weights):
    """
    reference: https://github.com/keras-team/keras/issues/2115
    :param y_true:
    :param y_pred:
    :param weights:
    :return:
    """
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])

    cross_ent = K.categorical_crossentropy(y_true, y_pred)
    return cross_ent * final_mask

# reference: https://github.com/GeekLiB/keras/tree/master/keras
def precision(y_true, y_pred):
    '''
    Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    '''
    Calculates the f-measure, the harmonic mean of precision and recall.
    '''
    return fbeta_score(y_true, y_pred, beta=1)


def train(train_sample_fn, val_sample_fn, model_out_root_dir,
          n_win_size=1000, n_feat=13, n_class=3,
          epochs=64, batch=1024, learn_rate=0.001, drop=0.5,
          fc_size=64, blocks='4_4_3', l2r=1e-4, temperature=5, lbl_smt_frac=0.1, pw=0, n_gpu=4, n_cpu=20):
    """

    :param train_sample_fn:
    :param val_sample_fn:
    :param model_out_root_dir:
    :param n_win_size:
    :param n_feat:
    :param n_class:
    :param epochs:
    :param batch:
    :param learn_rate:
    :param drop:
    :param fc_size:
    :param blocks:
    :param n_gpu:
    :param n_cpu:
    :param l2r:
    :param temperature:
    :param pw:
    :return:
    """

    _blocks = tuple(int(x) for x in blocks.split('_'))

    def out_name():
        str_blocks = [str(x) for x in blocks.split('_')]
        str_blk = ''.join(str_blocks)
        return 'b{0}_ep{1}_lr{2:.3f}_dr{3:.1f}_fc{4}_' \
               'blk{5}_win{6}_l2r{7}_temp{8}_lblsmt{9}_pw{10}'.format(batch,
                                                                      epochs,
                                                                      learn_rate,
                                                                      drop,
                                                                      fc_size,
                                                                      str_blk,
                                                                      n_win_size,
                                                                      str(l2r),
                                                                      temperature,
                                                                      int(
                                                                          lbl_smt_frac) if lbl_smt_frac == 0 else lbl_smt_frac,
                                                                      pw)

    train_sample_df = pd.read_csv(train_sample_fn, sep='\t')
    val_sample_df = pd.read_csv(val_sample_fn, sep='\t')

    # using np.array instead of pd.DataFrame to avoid possible deadlock in fit_genertaor()
    # ref: https://github.com/keras-team/keras/issues/10340
    train_sample_arr = train_sample_df.values
    val_sample_arr = val_sample_df.values

    del train_sample_df, val_sample_df

    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_name = 'cnvnet'
    base_model = cnv_net(n_win_size, n_feat, n_class, filters=32, kernel_size=16, strides=1, pool_size=2,
                         pool_stride=2, drop=drop, blocks=_blocks, fc_size=fc_size,
                         kernel_regular_l2=l2r, temperature=temperature, m_name=model_name)

    # base_model = cnv_net_seq(n_win_size, n_feat, n_class)
    if n_gpu > 1:
        model = multi_gpu_model(base_model, n_gpu)
    else:
        model = base_model

    _model_dir = os.path.join(model_out_root_dir, 'model_weight')
    if not os.path.isdir(_model_dir):
        os.mkdir(_model_dir)
    _tb_dir = os.path.join(model_out_root_dir, 'tb_logs')
    if not os.path.isdir(_tb_dir):
        os.mkdir(_tb_dir)
    _csvlogger_dir = os.path.join(model_out_root_dir, 'model_csvlogger')
    if not os.path.isdir(_csvlogger_dir):
        os.mkdir(_csvlogger_dir)

    model_fn = os.path.join(_model_dir, '{}-{}.hdf5'.format(out_name(), model_name))
    if os.path.exists(model_fn):
        os.remove(model_fn)

    tensorboard_fn = os.path.join(_tb_dir, '{}-{}'.format(out_name(), model_name))
    if os.path.isdir(tensorboard_fn):
        shutil.rmtree(tensorboard_fn, ignore_errors=True)

    csvlogger_fn = os.path.join(_csvlogger_dir, '{}-{}'.format(out_name(), model_name))
    if os.path.exists(csvlogger_fn):
        os.remove(csvlogger_fn)

    callbacks = [
        # Early stopping definition
        EarlyStopping(monitor='val_acc', patience=10, verbose=1),
        # Decrease learning rate by 0.5 factor
        AdvancedLearnignRateScheduler(monitor='val_acc', patience=1, verbose=1, mode='auto', decayRatio=0.5),
        # Saving best model
        MultiGPUCheckpointCallback(model_fn, base_model=base_model, monitor='val_acc',
                                   save_best_only=True, verbose=1, save_weights_only=True),
        # histogram_freq=0 because
        # ValueError: If printing histograms, validation_data must be provided, and cannot be a generator
        # set histogram_freq=0 to solve the problem
        # TensorBoard(tensorboard_fn, batch_size=batch, histogram_freq=0),
        CSVLogger(csvlogger_fn)
    ]

    # fine tune the cost function so that missclassification is weighted some how
    p_misclass_weight = np.ones((n_class, n_class))
    # there could be improved
    # penalizing FN
    p_misclass_weight[:, 0] = 2.0
    # penalizing FP
    p_misclass_weight[0, :] = 2.0
    p_misclass_weight[1, 2] = 2.0
    p_misclass_weight[2, 1] = 2.0
    p_misclass_weight[0, 0] = 1.0

    p_misclass_weight2 = np.ones((n_class, n_class))
    # there could be improved
    # penalizing FN
    p_misclass_weight2[:, 0] = 2.0
    # penalizing FP
    p_misclass_weight2[0, :] = 1.5
    p_misclass_weight2[0, 0] = 1.0

    misclass_dict = {1: p_misclass_weight, 2: p_misclass_weight2}

    # using weight loss funtion
    # https://github.com/keras-team/keras/issues/2115
    if pw > 0:
        custom_loss = partial(w_categorical_crossentropy, weights=misclass_dict[pw])
        custom_loss.__name__ = 'w_categorical_crossentropy'
    elif pw == 0:
        custom_loss = 'categorical_crossentropy'
    elif pw == -1:
        custom_loss = focal_loss(alpha=1.0)
    elif pw == -2:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.2, beta_t=0.8, gamma_f=2., alpha_f=1.)
    elif pw == -3:
        custom_loss = combine_weighted_ce_focal_loss(weights=misclass_dict[1], gamma_f=2., alpha_f=1.)
    elif pw == -4:
        custom_loss = combine_tversky_focal_loss(alpha_t=1, beta_t=1, gamma_f=2., alpha_f=1.)
    elif pw == -5:
        custom_loss = tversky_loss(alpha=0.5, beta=0.5)
    elif pw == -6:
        custom_loss = tversky_loss(alpha=0.8, beta=0.2)
    elif pw == -7:
        custom_loss = tversky_loss(alpha=0.1, beta=0.9)
    elif pw == -8:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.8, beta_t=0.2, gamma_f=2., alpha_f=1.)
    elif pw == -9:
        custom_loss = combine_tversky_focal_loss(alpha_t=4.0, beta_t=2.0, gamma_f=2., alpha_f=1.)
    elif pw == -10:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.8, beta_t=0.2, gamma_f=4., alpha_f=1.)
    elif pw == -11:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.8, beta_t=0.2, gamma_f=1.5, alpha_f=1.)
    elif pw == -12:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.7, beta_t=0.3, gamma_f=2, alpha_f=1.)
    elif pw == -13:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.8, beta_t=0.2, gamma_f=2.5, alpha_f=1.)
    elif pw == -14:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.8, beta_t=0.2, gamma_f=3, alpha_f=1.)
    elif pw == -15:  # call
        custom_loss = combine_tversky_focal_loss(alpha_t=0.9, beta_t=0.1, gamma_f=2.0, alpha_f=1.)
    elif pw == -16:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.6, beta_t=0.4, gamma_f=2.0, alpha_f=1.)
    elif pw == -17:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.7, beta_t=0.3, gamma_f=1.5, alpha_f=1.)
    elif pw == -18:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.6, beta_t=0.4, gamma_f=1.5, alpha_f=1.)

    model.compile(optimizer=keras.optimizers.Adam(lr=learn_rate),
                  loss=custom_loss,
                  metrics=['accuracy'])

    # model.compile(optimizer=keras.optimizers.Adam(lr=learn_rate),
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    training_batch_generator = CNVDataGenerator(train_sample_arr, batch,
                                                win_size=n_win_size, n_feat=n_feat, shuffle=True,
                                                smooth_factor=lbl_smt_frac)
    val_batch_generator = CNVDataGenerator(val_sample_arr, batch,
                                           win_size=n_win_size, n_feat=n_feat, shuffle=True, smooth_factor=lbl_smt_frac)

    # fit_generator
    model.fit_generator(generator=training_batch_generator,
                        steps_per_epoch=(len(train_sample_arr) // batch),
                        epochs=epochs,
                        verbose=1,
                        # class_weight=None,  # class_weight=class_weights
                        callbacks=callbacks,
                        validation_data=val_batch_generator,
                        validation_steps=(len(val_sample_arr) // batch),
                        use_multiprocessing=True,  # True
                        workers=n_cpu,  # n_cpu
                        max_queue_size=20)  # 80


def train2(target_train_h5_fn, target_val_h5_fn, model_out_root_dir,
           n_win_size=5000, n_feat=13, n_class=3, filters=32, kernel_size=16,
           epochs=64, batch=1024, learn_rate=0.001, drop=0.5,
           fc_size=64, blocks='4_4_3', l2r=1e-4, temperature=5, lbl_smt_frac=0.1, pw=0, n_gpu=4, n_cpu=20):

    _blocks = tuple(int(x) for x in blocks.split('_'))

    def out_name():
        str_blocks = [str(x) for x in blocks.split('_')]
        str_blk = ''.join(str_blocks)
        return 'b{0}_ep{1}_lr{2:.3f}_dr{3:.1f}_fc{4}_' \
               'blk{5}_win{6}_l2r{7}_temp{8}_lblsmt{9}_pw{10}'.format(batch,
                                                                      epochs,
                                                                      learn_rate,
                                                                      drop,
                                                                      fc_size,
                                                                      str_blk,
                                                                      n_win_size,
                                                                      str(l2r),
                                                                      temperature,
                                                                      int(
                                                                          lbl_smt_frac) if lbl_smt_frac == 0 else lbl_smt_frac,
                                                                      pw)

    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_name = 'cnvnet'
    base_model = cnv_net(n_win_size, n_feat, n_class, filters=filters, kernel_size=kernel_size, strides=1, pool_size=2,
                         pool_stride=2, drop=drop, blocks=_blocks, fc_size=fc_size,
                         kernel_regular_l2=l2r, temperature=temperature, m_name=model_name)

    # base_model = cnv_net_seq(n_win_size, n_feat, n_class)
    if n_gpu > 1:
        model = multi_gpu_model(base_model, n_gpu)
    else:
        model = base_model

    _model_dir = os.path.join(model_out_root_dir, 'model_weight')
    if not os.path.isdir(_model_dir):
        os.mkdir(_model_dir)
    _tb_dir = os.path.join(model_out_root_dir, 'tb_logs')
    if not os.path.isdir(_tb_dir):
        os.mkdir(_tb_dir)
    _csvlogger_dir = os.path.join(model_out_root_dir, 'model_csvlogger')
    if not os.path.isdir(_csvlogger_dir):
        os.mkdir(_csvlogger_dir)

    model_fn = os.path.join(_model_dir, '{}-{}.hdf5'.format(out_name(), model_name))
    if os.path.exists(model_fn):
        os.remove(model_fn)

    tensorboard_fn = os.path.join(_tb_dir, '{}-{}'.format(out_name(), model_name))
    if os.path.isdir(tensorboard_fn):
        shutil.rmtree(tensorboard_fn, ignore_errors=True)

    csvlogger_fn = os.path.join(_csvlogger_dir, '{}-{}'.format(out_name(), model_name))
    if os.path.exists(csvlogger_fn):
        os.remove(csvlogger_fn)

    with h5py.File(target_train_h5_fn, 'r') as train_h5:
        train_len = train_h5['x'].shape[0]
    with h5py.File(target_val_h5_fn, 'r') as val_h5:
        val_len = val_h5['x'].shape[0]

    callbacks = [
        # Early stopping definition
        EarlyStopping(monitor='val_acc', patience=10, verbose=1),
        # Decrease learning rate by 0.5 factor
        AdvancedLearnignRateScheduler(monitor='val_acc', patience=1, verbose=1, mode='auto', decayRatio=0.5),
        # CyclicLR(mode='triangular', base_lr=learn_rate, max_lr=0.1, step_size=6 * (train_len // batch)),
        # Saving best model
        MultiGPUCheckpointCallback(model_fn, base_model=base_model, monitor='val_acc',
                                   save_best_only=True, verbose=1, save_weights_only=True),
        # histogram_freq=0 because
        # ValueError: If printing histograms, validation_data must be provided, and cannot be a generator
        # set histogram_freq=0 to solve the problem
        # TensorBoard(tensorboard_fn, batch_size=batch, histogram_freq=0),
        CSVLogger(csvlogger_fn)
    ]

    # fine tune the cost function so that missclassification is weighted some how
    p_misclass_weight = np.ones((n_class, n_class))
    # there could be improved
    # penalizing FN
    p_misclass_weight[:, 0] = 2.0
    # penalizing FP
    p_misclass_weight[0, :] = 2.0
    p_misclass_weight[1, 2] = 2.0
    p_misclass_weight[2, 1] = 2.0
    p_misclass_weight[0, 0] = 1.0

    p_misclass_weight2 = np.ones((n_class, n_class))
    # there could be improved
    # penalizing FN
    p_misclass_weight2[:, 0] = 2.0
    # penalizing FP
    p_misclass_weight2[0, :] = 1.5
    p_misclass_weight2[0, 0] = 1.0

    misclass_dict = {1: p_misclass_weight, 2: p_misclass_weight2}

    # using weight loss funtion
    # https://github.com/keras-team/keras/issues/2115
    if pw > 0:
        custom_loss = partial(w_categorical_crossentropy, weights=misclass_dict[pw])
        custom_loss.__name__ = 'w_categorical_crossentropy'
    elif pw == 0:
        custom_loss = 'categorical_crossentropy'
    elif pw == -1:
        custom_loss = focal_loss(alpha=1.0)
    elif pw == -2:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.2, beta_t=0.8, gamma_f=2., alpha_f=1.)
    elif pw == -3:
        custom_loss = combine_weighted_ce_focal_loss(weights=misclass_dict[1], gamma_f=2., alpha_f=1.)
    elif pw == -4:
        custom_loss = combine_tversky_focal_loss(alpha_t=1, beta_t=1, gamma_f=2., alpha_f=1.)
    elif pw == -5:
        custom_loss = tversky_loss(alpha=0.5, beta=0.5)
    elif pw == -6:
        custom_loss = tversky_loss(alpha=0.8, beta=0.2)
    elif pw == -7:
        custom_loss = tversky_loss(alpha=0.1, beta=0.9)
    elif pw == -8:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.8, beta_t=0.2, gamma_f=2., alpha_f=1.)
        custom_loss.__name__ = 'tversky_focal_loss'
    elif pw == -9:
        custom_loss = combine_tversky_focal_loss(alpha_t=4.0, beta_t=2.0, gamma_f=2., alpha_f=1.)
    elif pw == -10:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.8, beta_t=0.2, gamma_f=4., alpha_f=1.)
    elif pw == -11:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.8, beta_t=0.2, gamma_f=1.5, alpha_f=1.)
    elif pw == -12:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.7, beta_t=0.3, gamma_f=2, alpha_f=1.)
    elif pw == -13:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.8, beta_t=0.2, gamma_f=2.5, alpha_f=1.)
    elif pw == -14:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.8, beta_t=0.2, gamma_f=3, alpha_f=1.)
    elif pw == -15:  # call
        custom_loss = combine_tversky_focal_loss(alpha_t=0.9, beta_t=0.1, gamma_f=2.0, alpha_f=1.)
    elif pw == -16:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.6, beta_t=0.4, gamma_f=2.0, alpha_f=1.)
    elif pw == -17:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.7, beta_t=0.3, gamma_f=1.5, alpha_f=1.)
    elif pw == -18:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.6, beta_t=0.4, gamma_f=1.5, alpha_f=1.)

    model.compile(optimizer=keras.optimizers.Adam(lr=learn_rate),
                  loss=custom_loss,
                  metrics=['accuracy'])

    # model.compile(optimizer=keras.optimizers.Adam(lr=learn_rate),
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    training_batch_generator = CNVDataGenerator3(target_train_h5_fn, train_len, batch,
                                                 win_size=n_win_size, n_feat=n_feat, smooth_factor=lbl_smt_frac)
    val_batch_generator = CNVDataGenerator3(target_val_h5_fn, val_len, batch,
                                            win_size=n_win_size, n_feat=n_feat, smooth_factor=lbl_smt_frac)

    # fit_generator
    model.fit_generator(generator=training_batch_generator,
                        steps_per_epoch=(train_len // batch),
                        epochs=epochs,
                        verbose=1,
                        # class_weight=None,  # class_weight=class_weights
                        callbacks=callbacks,
                        validation_data=val_batch_generator,
                        validation_steps=(val_len // batch),
                        use_multiprocessing=True,  # True
                        workers=n_cpu,  # n_cpu
                        max_queue_size=(n_cpu + 10))  # 80


def train3(target_train_h5_fn, target_val_h5_fn, model_out_root_dir,
           n_win_size=5000, n_feat=13, n_class=3, filters=32, kernel_size=16,
           epochs=64, batch=1024, learn_rate=0.001, drop=0.5,
           fc_size=64, blocks='4_4_3', l2r=1e-4, temperature=5, lbl_smt_frac=0.1, pw=0, n_gpu=4, n_cpu=20):

    _blocks = tuple(int(x) for x in blocks.split('_'))

    def out_name():
        str_blocks = [str(x) for x in blocks.split('_')]
        str_blk = ''.join(str_blocks)
        return 'b{0}_ep{1}_lr{2:.3f}_dr{3:.1f}_fc{4}_' \
               'blk{5}_win{6}_l2r{7}_temp{8}_lblsmt{9}_pw{10}'.format(batch,
                                                                      epochs,
                                                                      learn_rate,
                                                                      drop,
                                                                      fc_size,
                                                                      str_blk,
                                                                      n_win_size,
                                                                      str(l2r),
                                                                      temperature,
                                                                      int(
                                                                          lbl_smt_frac) if lbl_smt_frac == 0 else lbl_smt_frac,
                                                                      pw)

    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_name = 'cnvnet'
    base_model = cnv_net(n_win_size, n_feat, n_class, filters=filters, kernel_size=kernel_size, strides=1, pool_size=2,
                         pool_stride=2, drop=drop, blocks=_blocks, fc_size=fc_size,
                         kernel_regular_l2=l2r, temperature=temperature, m_name=model_name)

    # base_model = cnv_net_seq(n_win_size, n_feat, n_class)
    if n_gpu > 1:
        model = multi_gpu_model(base_model, n_gpu)
    else:
        model = base_model

    _model_dir = os.path.join(model_out_root_dir, 'model_weight')
    if not os.path.isdir(_model_dir):
        os.mkdir(_model_dir)
    _tb_dir = os.path.join(model_out_root_dir, 'tb_logs')
    if not os.path.isdir(_tb_dir):
        os.mkdir(_tb_dir)
    _csvlogger_dir = os.path.join(model_out_root_dir, 'model_csvlogger')
    if not os.path.isdir(_csvlogger_dir):
        os.mkdir(_csvlogger_dir)

    model_fn = os.path.join(_model_dir, '{}-{}.hdf5'.format(out_name(), model_name))
    if os.path.exists(model_fn):
        os.remove(model_fn)

    tensorboard_fn = os.path.join(_tb_dir, '{}-{}'.format(out_name(), model_name))
    if os.path.isdir(tensorboard_fn):
        shutil.rmtree(tensorboard_fn, ignore_errors=True)

    csvlogger_fn = os.path.join(_csvlogger_dir, '{}-{}'.format(out_name(), model_name))
    if os.path.exists(csvlogger_fn):
        os.remove(csvlogger_fn)

    with h5py.File(target_train_h5_fn, 'r') as train_h5:
        train_len = train_h5['x'].shape[0]
    with h5py.File(target_val_h5_fn, 'r') as val_h5:
        val_len = val_h5['x'].shape[0]

    callbacks = [
        # Early stopping definition
        EarlyStopping(monitor='val_fmeasure', mode='max', patience=10, verbose=1),
        # Decrease learning rate by 0.5 factor
        AdvancedLearnignRateScheduler(monitor='val_fmeasure', patience=1, verbose=1, mode='max', decayRatio=0.5),
        # CyclicLR(mode='triangular', base_lr=learn_rate, max_lr=0.1, step_size=6 * (train_len // batch)),
        # Saving best model
        MultiGPUCheckpointCallback(model_fn, base_model=base_model, monitor='val_fmeasure', mode='max',
                                   save_best_only=True, verbose=1, save_weights_only=True),
        # histogram_freq=0 because
        # ValueError: If printing histograms, validation_data must be provided, and cannot be a generator
        # set histogram_freq=0 to solve the problem
        # TensorBoard(tensorboard_fn, batch_size=batch, histogram_freq=0),
        CSVLogger(csvlogger_fn)
    ]

    # fine tune the cost function so that missclassification is weighted some how
    p_misclass_weight = np.ones((n_class, n_class))
    # there could be improved
    # penalizing FN
    p_misclass_weight[:, 0] = 2.0
    # penalizing FP
    p_misclass_weight[0, :] = 2.0
    p_misclass_weight[1, 2] = 2.0
    p_misclass_weight[2, 1] = 2.0
    p_misclass_weight[0, 0] = 1.0

    p_misclass_weight2 = np.ones((n_class, n_class))
    # there could be improved
    # penalizing FN
    p_misclass_weight2[:, 0] = 2.0
    # penalizing FP
    p_misclass_weight2[0, :] = 1.5
    p_misclass_weight2[0, 0] = 1.0

    misclass_dict = {1: p_misclass_weight, 2: p_misclass_weight2}

    # using weight loss funtion
    # https://github.com/keras-team/keras/issues/2115
    if pw > 0:
        custom_loss = partial(w_categorical_crossentropy, weights=misclass_dict[pw])
        custom_loss.__name__ = 'w_categorical_crossentropy'
    elif pw == 0:
        custom_loss = 'categorical_crossentropy'
    elif pw == -1:
        custom_loss = focal_loss(alpha=1.0)
    elif pw == -2:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.2, beta_t=0.8, gamma_f=2., alpha_f=1.)
    elif pw == -3:
        custom_loss = combine_weighted_ce_focal_loss(weights=misclass_dict[1], gamma_f=2., alpha_f=1.)
    elif pw == -4:
        custom_loss = combine_tversky_focal_loss(alpha_t=1, beta_t=1, gamma_f=2., alpha_f=1.)
    elif pw == -5:
        custom_loss = tversky_loss(alpha=0.5, beta=0.5)
    elif pw == -6:
        custom_loss = tversky_loss(alpha=0.8, beta=0.2)
    elif pw == -7:
        custom_loss = tversky_loss(alpha=0.1, beta=0.9)
    elif pw == -8:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.8, beta_t=0.2, gamma_f=2., alpha_f=1.)
        custom_loss.__name__ = 'tversky_focal_loss'
    elif pw == -9:
        custom_loss = combine_tversky_focal_loss(alpha_t=4.0, beta_t=2.0, gamma_f=2., alpha_f=1.)
    elif pw == -10:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.8, beta_t=0.2, gamma_f=4., alpha_f=1.)
    elif pw == -11:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.8, beta_t=0.2, gamma_f=1.5, alpha_f=1.)
    elif pw == -12:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.7, beta_t=0.3, gamma_f=2, alpha_f=1.)
    elif pw == -13:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.8, beta_t=0.2, gamma_f=2.5, alpha_f=1.)
    elif pw == -14:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.8, beta_t=0.2, gamma_f=3, alpha_f=1.)
    elif pw == -15:  # call
        custom_loss = combine_tversky_focal_loss(alpha_t=0.9, beta_t=0.1, gamma_f=2.0, alpha_f=1.)
    elif pw == -16:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.6, beta_t=0.4, gamma_f=2.0, alpha_f=1.)
    elif pw == -17:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.7, beta_t=0.3, gamma_f=1.5, alpha_f=1.)
    elif pw == -18:
        custom_loss = combine_tversky_focal_loss(alpha_t=0.6, beta_t=0.4, gamma_f=1.5, alpha_f=1.)

    model.compile(optimizer=keras.optimizers.Adam(lr=learn_rate),
                  loss=custom_loss,
                  metrics=['accuracy', precision, recall, fmeasure])

    # model.compile(optimizer=keras.optimizers.Adam(lr=learn_rate),
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    training_batch_generator = CNVDataGenerator3(target_train_h5_fn, train_len, batch,
                                                 win_size=n_win_size, n_feat=n_feat, smooth_factor=lbl_smt_frac)
    val_batch_generator = CNVDataGenerator3(target_val_h5_fn, val_len, batch,
                                            win_size=n_win_size, n_feat=n_feat, smooth_factor=lbl_smt_frac)

    # fit_generator
    model.fit_generator(generator=training_batch_generator,
                        steps_per_epoch=(train_len // batch),
                        epochs=epochs,
                        verbose=1,
                        # class_weight=None,  # class_weight=class_weights
                        callbacks=callbacks,
                        validation_data=val_batch_generator,
                        validation_steps=(val_len // batch),
                        use_multiprocessing=True,  # True
                        workers=n_cpu,  # n_cpu
                        max_queue_size=(n_cpu + 10))  # 80


def train_ae(target_train_h5_fn, target_val_h5_fn, model_out_root_dir,
             n_win_size=10000, n_feat=9,
             epochs=64, batch=128, learn_rate=0.001, drop=0.5, l2r=1e-4,
             n_gpu=4, n_cpu=20):

    n_res_blks = -1
    t_win_size = n_win_size
    while t_win_size % 2 == 0:
        n_res_blks += 1
        t_win_size = int(t_win_size // 2)

    n_res_blks = max(0, n_res_blks)

    def out_name():
        return 'b{0}_ep{1}_lr{2:.3f}_dr{3:.1f}_win{4}_l2r{5}_resnlk{6}'.format(batch, epochs, learn_rate,
                                                                               drop,
                                                                               n_win_size,
                                                                               str(l2r),
                                                                               n_res_blks)

    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_name = 'cnv_aenet'
    base_model = cnv_aenet(n_win_size, n_feat, drop=drop, n_res_blks=n_res_blks,
                           kernel_regular_l2=l2r, m_name=model_name)
    if n_gpu > 1:
        model = multi_gpu_model(base_model, n_gpu)
    else:
        model = base_model

    _model_dir = os.path.join(model_out_root_dir, 'model_weight')
    if not os.path.isdir(_model_dir):
        os.mkdir(_model_dir)
    _tb_dir = os.path.join(model_out_root_dir, 'tb_logs')
    if not os.path.isdir(_tb_dir):
        os.mkdir(_tb_dir)
    _csvlogger_dir = os.path.join(model_out_root_dir, 'model_csvlogger')
    if not os.path.isdir(_csvlogger_dir):
        os.mkdir(_csvlogger_dir)

    model_fn = os.path.join(_model_dir, '{}-{}.hdf5'.format(out_name(), model_name))
    if os.path.exists(model_fn):
        os.remove(model_fn)

    tensorboard_fn = os.path.join(_tb_dir, '{}-{}'.format(out_name(), model_name))
    if os.path.isdir(tensorboard_fn):
        shutil.rmtree(tensorboard_fn, ignore_errors=True)

    csvlogger_fn = os.path.join(_csvlogger_dir, '{}-{}'.format(out_name(), model_name))
    if os.path.exists(csvlogger_fn):
        os.remove(csvlogger_fn)

    with h5py.File(target_train_h5_fn, 'r') as train_h5:
        train_len = train_h5['x'].shape[0]
    with h5py.File(target_val_h5_fn, 'r') as val_h5:
        val_len = val_h5['x'].shape[0]

    callbacks = [
        # Early stopping definition
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        # Decrease learning rate by 0.5 factor
        AdvancedLearnignRateScheduler(monitor='val_loss', patience=2, verbose=1, mode='auto', decayRatio=0.5),
        # CyclicLR(mode='triangular', base_lr=learn_rate, max_lr=0.1, step_size=6 * (train_len // batch)),
        # Saving best model
        MultiGPUCheckpointCallback(model_fn, base_model=base_model, monitor='val_loss',
                                   save_best_only=True, verbose=1, save_weights_only=True),
        # histogram_freq=0 because
        # ValueError: If printing histograms, validation_data must be provided, and cannot be a generator
        # set histogram_freq=0 to solve the problem
        # TensorBoard(tensorboard_fn, batch_size=batch, histogram_freq=0),
        CSVLogger(csvlogger_fn)
    ]

    model.compile(optimizer=keras.optimizers.Adam(lr=learn_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    training_batch_generator = CNVDataGenerator_AE(target_train_h5_fn, train_len, batch,
                                                   win_size=n_win_size, n_feat=n_feat, smooth_factor=0)
    val_batch_generator = CNVDataGenerator_AE(target_val_h5_fn, val_len, batch,
                                              win_size=n_win_size, n_feat=n_feat, smooth_factor=0)

    # fit_generator
    model.fit_generator(generator=training_batch_generator,
                        steps_per_epoch=(train_len // batch),
                        epochs=epochs,
                        verbose=1,
                        # class_weight=None,  # class_weight=class_weights
                        callbacks=callbacks,
                        validation_data=val_batch_generator,
                        validation_steps=(val_len // batch),
                        use_multiprocessing=True,  # True
                        workers=n_cpu,  # n_cpu
                        max_queue_size=(n_cpu + 10))  # 80
