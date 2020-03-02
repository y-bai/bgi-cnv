#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: model_crt_1dcnn_resnet.py
    Description:
    
Created by Yong Bai on 2019/7/30 11:18 AM.
"""
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Flatten, Dropout, MaxPooling1D, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, warnings, TensorBoard, CSVLogger
from keras import backend as K
from keras.utils import multi_gpu_model

from cnv_utils import AdvancedLearnignRateScheduler
from cnv_utils import MultiGPUCheckpointCallback

import tensorflow as tf
import gc
import os
from sklearn.model_selection import train_test_split

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def conv1d_resnet(win_size, n_in_feat, n_out_class):
    k = 1  # increment every 4th residual block
    p = True  # pool toggle every other residual block (end with 2^8)

    n_filter = 64
    filter_size = 16
    filter_stride = 1
    pool_size = 2
    pool_stride = 2

    drop = 0.5

    input_x = Input(shape=(win_size, n_in_feat), name='input')

    # first conv block
    x = Conv1D(filters=n_filter, kernel_size=filter_size, padding='same', strides=filter_stride,
               kernel_initializer='he_normal')(input_x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second conv block with residual net
    # left branch
    x1 = Conv1D(filters=n_filter, kernel_size=filter_size, padding='same', strides=filter_stride,
               kernel_initializer='he_normal')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(drop)(x1)
    x1 = Conv1D(filters=n_filter, kernel_size=filter_size, padding='same', strides=filter_stride,
                kernel_initializer='he_normal')(x1)
    x1 = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(x1)

    # right branch
    x2 = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(x)


    # merge both branch
    x = keras.layers.add([x1, x2])
    del x1, x2
    # main loop
    p = not p
    for l in range(15):

        if (l % 4 == 0) and (l > 0):  # increment k on every fourth residual block
            k += 1
            # increase depth by 1x1 Convolution case dimension shall change
            xshort = Conv1D(filters=n_filter * k, kernel_size=1)(x)
        else:
            xshort = x

        # Left branch (convolutions)
        # notice the ordering of the operations has changed
        x1 = BatchNormalization()(x)
        x1 = Activation('relu')(x1)
        x1 = Dropout(drop)(x1)
        x1 = Conv1D(filters=n_filter * k,
                    kernel_size=filter_size,
                    padding='same',
                    strides=filter_stride,
                    kernel_initializer='he_normal')(x1)

        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Dropout(drop)(x1)
        x1 = Conv1D(filters=n_filter * k,
                    kernel_size=filter_size,
                    padding='same',
                    strides=filter_stride,
                    kernel_initializer='he_normal')(x1)
        if p:
            x1 = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(x1)

        # Right branch: shortcut connection
        if p:
            x2 = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(xshort)
        else:
            x2 = xshort  # pool or identity
        # Merging branches
        x = keras.layers.add([x1, x2])
        # change parameters
        p = not p  # toggle pooling

    # Final bit
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(1000)(x)
    x = Dense(1000)(x)
    out = Dense(n_out_class, activation='softmax')(x)
    model = Model(inputs=input_x, outputs=out)
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    #model.summary()
    #sequential_model_to_ascii_printout(model)
    # plot_model(model, to_file='model.png')
    return model


def model_evaluate(model_root_dir, total_samples_ls_fn, test_ratio,
                   win_size, min_size, min_r, min_f_deldup, min_f_neu, is_norm):
    """

    :param model_root_dir:
    :param total_samples_ls_fn:
    :param test_ratio:
    :param win_size:
    :param min_size:
    :param min_r:
    :param min_f_deldup:
    :param min_f_neu:
    :param is_norm:
    :return:
    """

    xtest = []
    ytest = []
    # load test data set
    test_set_fn = os.path.join(model_root_dir,
                               'test{:.2f}_win{}_minsize{}_dataset.npz'.format(test_ratio, win_size, min_size))
    if os.path.exists(test_set_fn):
        with np.load(test_set_fn) as test_set:
            xtest = test_set['x_test']
            ytest = test_set['y_test']
    else:
        from keras.utils import to_categorical
        _, test_ids = get_train_test_ids(model_root_dir, total_samples_ls_fn, test_ratio)
        x_test = []
        y_test_= []

        for ix, test_id in enumerate(test_ids):
            x, y = loaddata(test_id, model_root_dir, win_size, min_r, min_f_deldup, min_f_neu, is_norm)

            if (not (x is None)) and (not (y is None)):
                if len(x_test) == 0:
                    x_test = x
                    y_test_ = y
                else:
                    x_test = np.concatenate((x_test, x), axis=0)
                    y_test_ = np.concatenate((y_test_, y))

        y_ = [1 if x == 'DEL' else 2 if x == 'DUP' else 0 for x in y_test_]
        y_test = to_categorical(y_)
        logger.info('x test: {}, y test: {}'.format(x_test.shape, y_test.shape))
        np.savez_compressed(test_set_fn, x_test=x_test, y_test=y_test)
        xtest = x_test
        ytest = y_test

    n_example, n_win_size, n_feat = xtest.shape
    n_class = ytest.shape[1]
    model = conv1d_resnet(n_win_size, n_feat, n_class)
    model.load_weights(os.path.join(model_root_dir, 'models/weights-best-multigpu_k0.hdf5'))
    ypred = model.predict(xtest)
    test_pred_fn = os.path.join(model_root_dir,
                                'models/test{:.2f}_win{}_minsize{}-multigpu_pred.npz'.format(
                                    test_ratio, win_size, min_size))
    np.savez_compressed(test_pred_fn, ypred=ypred, ytrue=ytest)
    logger.info('Predict Done, result saved at {}'.format(test_pred_fn))


def multi_gpu_model_train(x_train, y_train, model_root_dir, epoches=20, batch=1024, k_fold=5, n_gpu=4):
    """

    :param x_train:
    :param y_train:
    :param model_root_dir:
    :param epoches:
    :param batch:
    :param k_fold:
    :param n_gpu:
    :return:
    """
    n_example, n_win_size, n_feat = x_train.shape
    n_class = y_train.shape[1]

    # cross validation runs
    for k in range(k_fold):
        if k > 0:
            break
        logger.info('>>>>>cross validation run {}'.format(k + 1))

        # split train and validation set from x_train, y_train
        k_x_train, k_x_val, k_y_train, k_y_val = train_test_split(x_train, y_train, test_size=1.0 / k_fold)

        logger.info('k_x_train: {}, k_x_val: {}, k_y_train: {}, k_y_val: {}'.format(
            k_x_train.shape, k_x_val.shape, k_y_train.shape, k_y_val.shape))

        # load model
        base_model = conv1d_resnet(n_win_size, n_feat, n_class)
        if n_gpu > 1:
            model = multi_gpu_model(base_model, n_gpu)
        else:
            model = base_model

        # Callbacks definition
        model_fn = os.path.join(model_root_dir, 'models/weights-best-multigpu_k{}.hdf5'.format(k))
        tensorboard_fn = os.path.join(model_root_dir, 'models/logs/tb-multigpu_k{}'.format(k))
        csvlogger_fn = os.path.join(model_root_dir, 'models/csvlogger-multigpu_k{}'.format(k))
        callbacks = [
            # Early stopping definition
            EarlyStopping(monitor='val_acc', patience=5, verbose=1),
            # Decrease learning rate by 0.1 factor
            AdvancedLearnignRateScheduler(monitor='val_acc', patience=1, verbose=1, mode='auto', decayRatio=0.8),
            # Saving best model
            MultiGPUCheckpointCallback(model_fn, base_model=base_model, monitor='val_acc',
                                       save_best_only=True, verbose=1, save_weights_only=True),
            TensorBoard(tensorboard_fn, batch_size=batch, histogram_freq=1),
            CSVLogger(csvlogger_fn)
        ]

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Train model
        model.fit(k_x_train, k_y_train, validation_data=(k_x_val, k_y_val),
                  epochs=epoches, batch_size=batch, callbacks=callbacks)
        #
        K.clear_session()
        gc.collect()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)


def model_train(x_train, y_train, model_root_dir, epoches=20, batch=1024, k_fold=5):

    n_example, n_win_size, n_feat = x_train.shape
    n_class = y_train.shape[1]

    # cross validation runs
    for k in range(k_fold):
        logger.info('>>>>>cross validation run {}'.format(k+1))

        # split train and validation set from x_train, y_train
        k_x_train, k_x_val, k_y_train, k_y_val = train_test_split(x_train, y_train, test_size=1.0/k_fold)

        logger.info('k_x_train: {}, k_x_val: {}, k_y_train: {}, k_y_val: {}'.format(
            k_x_train.shape, k_x_val.shape, k_y_train.shape, k_y_val.shape))

        # Callbacks definition
        model_fn = os.path.join(model_root_dir, 'models/weights-best_k{}.hdf5'.format(k))
        callbacks = [
            # Early stopping definition
            EarlyStopping(monitor='val_acc', patience=5, verbose=1),
            # Decrease learning rate by 0.1 factor
            AdvancedLearnignRateScheduler(monitor='val_acc', patience=1, verbose=1, mode='auto', decayRatio=0.8),
            # Saving best model
            ModelCheckpoint(model_fn, monitor='val_acc', save_best_only=True, verbose=1),
        ]
        # load model
        model = conv1d_resnet(n_win_size, n_feat, n_class)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Train model
        model.fit(k_x_train, k_y_train, validation_data=(k_x_val, k_y_val),
                  epochs=epoches, batch_size=batch, callbacks=callbacks)
        #
        K.clear_session()
        gc.collect()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)


def loaddata(sample_id, model_root_dir, win_size=1000, min_r=0.1, min_f1=0.01, min_f2=-2, is_norm=False):
    """

    :param sample_id:
    :param model_root_dir:
    :param win_size: window size
    :param min_r: minimum value of ratio of base coverage along thr window
    :param min_f1: minimum frequency for DEL or DUP data, default 0.01
    :param min_f2: minimum frequency for NEU data. The value is just only NEU
    :param is_norm: this is the same as the value when creating training dataset in mat_crt_train_data
    :return:
    """
    logger.info('Loading dataset for sample {}...'.format(sample_id))
    data_dir = os.path.join(model_root_dir, 'data4models')
    del_dup_data_fn = os.path.join(data_dir, "ds_data4model_{0}_{1}_{2:.2f}_{3:.2f}_{4}.npz".format(
        sample_id, win_size, min_r, min_f1, is_norm))
    neu_data_fn = os.path.join(data_dir, "ds_data4model_{0}_{1}_{2:.2f}_{3:.2f}_{4}.npz".format(
        sample_id, win_size, min_r, min_f2, is_norm))

    if os.path.exists(del_dup_data_fn) and os.path.exists(neu_data_fn):
        with np.load(del_dup_data_fn) as del_dup_data, np.load(neu_data_fn) as neu_data:
            x = np.concatenate((del_dup_data['x'], neu_data['x']), axis=0)
            y = np.concatenate((del_dup_data['y'], neu_data['y']))
        return x, y

    else:
        logger.warning('sample {} does not have enough data.'.format(sample_id))
        return None, None


def get_train_test_ids(model_root_dir, total_samples_ls_fn, test_size=0.3):
    """

    :param model_root_dir:
    :param total_samples_ls_fn:
    :param test_size:
    :return:
    """

    # create the train and test sample ids
    logger.info("Loading training and testing sample ids...")
    train_test_sample_list = os.path.join(model_root_dir, '1k_train_test{:.2f}_sample_ids.npz'.format(test_size))
    if not os.path.exists(train_test_sample_list):
        if not os.path.exists(total_samples_ls_fn):
            raise FileNotFoundError('sample list file does not exist. {}'.format(total_samples_ls_fn))
        else:
            sample_id_map = np.loadtxt(total_samples_ls_fn, delimiter='\t', usecols=(0, 1), dtype='str')
            sample_id_arr = sample_id_map[:, 0]
            train_ids, test_ids = train_test_split(sample_id_arr, test_size=test_size, random_state=123)
            np.savez(train_test_sample_list, train_ids=train_ids, test_ids=test_ids)
            return train_ids, test_ids
    else:
        with np.load(train_test_sample_list) as sample_ids:
            return sample_ids['train_ids'], sample_ids['test_ids']


def build_train_set(model_root_dir, total_samples_ls_fn, win_size,
                    min_r, min_f_deldup, min_f_neu, is_norm, test_ratio=0.3, min_size=100000):
    """

    :param model_root_dir:
    :param total_samples_ls_fn:
    :param win_size:
    :param min_r:
    :param min_f_deldup:
    :param min_f_neu:
    :param is_norm:
    :param min_sze:
    :return:
    """

    train_set_fn = os.path.join(model_root_dir,
                                'train{:.2f}_win{}_minsize{}_dataset.npz'.format(1-test_ratio, win_size, min_size))
    if os.path.exists(train_set_fn):
        with np.load(train_set_fn) as train_set:
            return train_set['x_train'], train_set['y_train']

    from keras.utils import to_categorical
    train_ids, _ = get_train_test_ids(model_root_dir, total_samples_ls_fn, test_ratio)
    x_train = []
    y_train_ = []
    for ix, train_id in enumerate(train_ids):
        x, y = loaddata(train_id, model_root_dir, win_size, min_r, min_f_deldup, min_f_neu, is_norm)

        if (not (x is None)) and (not (y is None)):

            # deal with data balance
            del_idx = np.where(y == 'DEL')[0]
            dup_idx = np.where(y == 'DUP')[0]
            neu_idx = np.where(y == 'NEU')[0]

            len_del_idx = len(del_idx)
            len_dup_idx = len(dup_idx)
            len_neu_idx = len(neu_idx)

            if len_del_idx == 0 or len_dup_idx == 0 or len_neu_idx == 0:
                logger.warning('del len: {}, dup len: {}, neu len: {}'.format(len_del_idx, len_dup_idx, len_neu_idx))
                continue

            logger.info('del len: {}, dup len: {}, neu len: {}'.format(len_del_idx, len_dup_idx, len_neu_idx))
            min_idx_len = np.min([len(del_idx), len(dup_idx), len(neu_idx)])

            f_del_idx = np.random.choice(del_idx, min_idx_len, replace=False)
            f_dup_idx = np.random.choice(dup_idx, min_idx_len, replace=False)
            f_neu_idx = np.random.choice(neu_idx, min_idx_len, replace=False)

            f_idx = np.concatenate((f_del_idx, f_dup_idx, f_neu_idx))
            y_ = y[f_idx]
            x_ = x[f_idx, :, :]

            if len(x_train) == 0:
                x_train = x_
                y_train_ = y_
            else:
                x_train = np.concatenate((x_train, x_), axis=0)
                y_train_ = np.concatenate((y_train_, y_))

            if len(y_train_) >= 3*min_size:
                break
    y = [1 if x == 'DEL' else 2 if x == 'DUP' else 0 for x in y_train_]
    y_train = to_categorical(y)
    del y_train_
    gc.collect()
    logger.info('x train: {}, y train: {}'.format(x_train.shape, y_train.shape))
    np.savez_compressed(train_set_fn, x_train=x_train, y_train=y_train)
    return x_train, y_train


if __name__ == "__main__":

    # path for save the data
    model_root_dir = "/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/1dcnn_resnet"
    total_samples_ls_fn = "/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/sample.list"

    win_size = 1000
    min_r = 0.1
    min_f_deldup = 0.01
    min_f_neu = -2
    is_norm = False

    test_ratio = 0.1

    example_min_size = 160000

    x_train, y_train = build_train_set(model_root_dir, total_samples_ls_fn,
                                       win_size, min_r, min_f_deldup, min_f_neu, is_norm, test_ratio, example_min_size)

    # single gpu train
    # model_train(x_train, y_train, model_root_dir)

    # multi gpu train
    multi_gpu_model_train(x_train, y_train, model_root_dir)

    # predict
    # model_evaluate(model_root_dir, total_samples_ls_fn, test_ratio,
    #                win_size, example_min_size, min_r, min_f_deldup, min_f_neu, is_norm)

    # model = conv1d_resnet(1000, 13, 3)
    # model.summary()


