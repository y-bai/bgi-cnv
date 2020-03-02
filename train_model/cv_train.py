#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: cv_train.py
    Description: cross validation by GP.
    will consider generator...
    
Created by Yong Bai on 2019/8/26 11:31 AM.
"""

import os
import gc
import numpy as np
from model import cnv_net

import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger
from model_utils import AdvancedLearnignRateScheduler, MultiGPUCheckpointCallback
from keras.utils import multi_gpu_model
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from skopt import gp_minimize, dump
from skopt.space import Categorical
from skopt.utils import use_named_args

best_accuracy = 0.0


def cv_train(x_train, y_train, model_root_dir, n_gpu=4, n_cpu=-1):
    """
    :param x_train:
    :param y_train:
    :param model_root_dir:
    :param n_gpu:
    :param n_cpu:
    :return:
    """

    K.clear_session()
    gc.collect()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    _, n_win_size, n_feat = x_train.shape
    n_class = y_train.shape[1]

    # search-dimension
    dim_nb_batchs = Categorical(categories=['128', '256', '512', '1024'], name='batch_size')
    dim_nb_epochs = Categorical(categories=['20', '30', '40', '50'], name='epoch')
    dim_lrs = Categorical(categories=['0.1', '0.01', '0.001', '0.0001'], name='learn_rate')
    dim_drops = Categorical(categories=['0.1', '0.2', '0.5'], name='drop')
    dim_fc_sizes = Categorical(categories=['32', '64', '128'], name='fc_size')
    dim_net_blocks = Categorical(categories=['4_3', '4_4_1', '4_4_3', '4_4_4_1'],
                                 name='blocks')

    search_dim = [dim_nb_batchs,
                  dim_nb_epochs,
                  dim_lrs,
                  dim_drops,
                  dim_fc_sizes,
                  dim_net_blocks]
    default_param = ['1024', '20', '0.01', '0.2', '32', '4_3']

    _model_dir = os.path.join(model_root_dir, 'model_cv/model_weight')
    if not os.path.isdir(_model_dir):
        os.mkdir(_model_dir)
    _tb_dir = os.path.join(model_root_dir, 'model_cv/logs')
    if not os.path.isdir(_tb_dir):
        os.mkdir(_tb_dir)
    _csvlogger_dir = os.path.join(model_root_dir, 'model_cv/model_metrics')
    if not os.path.isdir(_csvlogger_dir):
        os.mkdir(_csvlogger_dir)

    def out_name(batch_size, epoch, learn_rate, drop, fc_size, blocks):
        str_blocks = [str(x) for x in blocks.split('_')]
        str_blk = ''.join(str_blocks)

        return 'b{0}_e{1}_lr{2:.3f}_dr{3:.1f}_fc{4}_blk{5}'.format(batch_size,
                                                                   epoch,
                                                                   learn_rate,
                                                                   drop,
                                                                   fc_size,
                                                                   str_blk,
                                                                   )

    # y_train_labels = np.argmax(y_train, axis=1)
    # skf = StratifiedKFold(n_splits=1, random_state=123, shuffle=True)

    @use_named_args(dimensions=search_dim)
    def gp_fitness(batch_size, epoch, learn_rate, drop, fc_size, blocks):

        batch_size = int(batch_size)
        epoch = int(epoch)
        learn_rate = float(learn_rate)
        drop = float(drop)
        fc_size = int(fc_size)

        print('batch_size: {}'.format(batch_size))
        print('epoch: {}'.format(epoch))
        print('learn rate: {0:.3f}'.format(learn_rate))
        print('drop ratio: {0:.1f}'.format(drop))
        print('fc size: {}'.format(fc_size))
        print('blocks: {}'.format(blocks))

        _blocks = (int(x) for x in blocks.split('_'))

        tmp_out_name = out_name(batch_size, epoch, learn_rate,
                                drop, fc_size, blocks)

        val_acc_arr = []

        # for i, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train_labels)):
        # ix_train1, ix_val1 = x_train[train_idx], x_train[val_idx]
        # iy_train1, iy_val1 = y_train[train_idx], y_train[val_idx]
        for i in range(1):
            ix_train1, ix_val1, iy_train1, iy_val1 = train_test_split(x_train, y_train, test_size=0.2,
                                                                      random_state=123)
            nb_trains = ix_train1.shape[0] // batch_size
            nb_examples = batch_size * nb_trains
            k_x_train = ix_train1[:nb_examples]
            k_y_train = iy_train1[:nb_examples]
            k_x_val = np.concatenate((ix_val1, ix_train1[nb_examples:]), axis=0)
            k_y_val = np.concatenate((iy_val1, iy_train1[nb_examples:]), axis=0)

            del ix_train1, ix_val1, iy_train1, iy_val1
            # gc.collect()

            model_fn = os.path.join(_model_dir, '{0}-k{1}.hdf5'.format(tmp_out_name, i))
            tensorboard_fn = os.path.join(_tb_dir, '{0}-tb_k{1}'.format(tmp_out_name, i))
            csvlogger_fn = os.path.join(_csvlogger_dir, '{0}-csvlogger_k{1}'.format(tmp_out_name, i))

            base_model = cnv_net(n_win_size, n_feat, n_class,
                                 filters=32, kernel_size=16, strides=1, pool_size=2,
                                 pool_stride=2, drop=drop, blocks=_blocks, fc_size=fc_size,
                                 kernel_regular_l2=None, m_name=tmp_out_name)
            if n_gpu > 1:
                model = multi_gpu_model(base_model, n_gpu)
            else:
                model = base_model

            callbacks = [
                # Early stopping definition
                EarlyStopping(monitor='val_acc', patience=5, verbose=1),
                # Decrease learning rate
                AdvancedLearnignRateScheduler(monitor='val_acc', patience=1, verbose=1, mode='auto',
                                              decayRatio=0.5),
                # Saving best model
                # MultiGPUCheckpointCallback(model_fn, base_model=base_model, monitor='val_acc',
                #                            save_best_only=True, verbose=1, save_weights_only=True),
                # TensorBoard(tensorboard_fn, batch_size=batch_size, histogram_freq=2),
                CSVLogger(csvlogger_fn)
            ]

            model.compile(optimizer=keras.optimizers.Adam(lr=learn_rate),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            hist = model.fit(k_x_train, k_y_train, validation_data=(k_x_val, k_y_val),
                             epochs=epoch, batch_size=batch_size, callbacks=callbacks)

            i_val_acc = hist.history['val_acc'][-1]
            print("Accuracy: {0:.6%}".format(i_val_acc))
            val_acc_arr.append(i_val_acc)

            del model
            del k_x_train, k_y_train, k_x_val, k_y_val

            K.clear_session()
            gc.collect()
            i_config = tf.ConfigProto()
            i_config.gpu_options.allow_growth = True
            # i_config = tf.ConfigProto(device_count={'GPU': n_gpu, 'CPU': n_cpu})
            i_sess = tf.Session(config=i_config)
            K.set_session(i_sess)

        cv_mean_val_acc = np.mean(val_acc_arr)

        global best_accuracy
        if cv_mean_val_acc > best_accuracy:
            best_accuracy = cv_mean_val_acc

        return -cv_mean_val_acc

    search_result = gp_minimize(func=gp_fitness,
                                dimensions=search_dim,
                                acq_func='EI',  # Expected Improvement.
                                n_calls=20,
                                n_jobs=n_cpu,
                                x0=default_param)
    del search_result.specs["args"]["func"]

    dump(search_result, os.path.join(model_root_dir, 'model_cv/gp_search_res.pickle'))
