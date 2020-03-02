#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: train_run.py
    Description:
    
Created by Yong Bai on 2019/8/20 2:38 PM.
"""
import os
import gc
import numpy as np
from model import cnv_net, cnv_simple_net, ResNet1D, cnv_net_seq
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger
from cnv_utils import AdvancedLearnignRateScheduler, MultiGPUCheckpointCallback
from keras.utils import multi_gpu_model
from keras import backend as K
from sklearn.model_selection import train_test_split


def train(training_generator, validation_data, model_root_dir, epochs=50, batch=256,
          learn_rate=0.001, drop=0.5, fc_size=128, blocks='4_4_3', n_gpu=4, n_cpu=10):
    """
    epoch from 50 -> 60
    :param x_train:
    :param y_train:
    :param model_root_dir:
    :param epochs:
    :param batch:
    :param learn_rate:
    :param drop:
    :param fc_size:
    :param blocks:
    :param n_gpu:
    :return:
    """
    # n_example, n_win_size, n_feat = x_train.shape
    n_win_size =1000
    n_class = 3
    n_feat=13
    # k_x_train1, k_x_val1, k_y_train1, k_y_val1 = train_test_split(x_train, y_train, test_size=0.2, random_state=123)

    # nb_trains = k_x_train1.shape[0] // batch
    # nb_examples = batch * nb_trains
    # k_x_train = k_x_train1[:nb_examples]
    # k_y_train = k_y_train1[:nb_examples]
    # k_x_val = np.concatenate((k_x_val1, k_x_train1[nb_examples:]), axis=0)
    # k_y_val = np.concatenate((k_y_val1, k_y_train1[nb_examples:]), axis=0)
    # del k_x_train1, k_x_val1, k_y_train1, k_y_val1
    gc.collect()

    def out_name():
        str_blocks = [str(x) for x in blocks.split('_')]
        str_blk = ''.join(str_blocks)
        return 'b{0}_e{1}_lr{2:.3f}_dr{3:.1f}_fc{4}_blk{5}_win{6}'.format(batch,
                                                                          epochs,
                                                                          learn_rate,
                                                                          drop,
                                                                          fc_size,
                                                                          str_blk,
                                                                          n_win_size)

    _blocks = (int(x) for x in blocks.split('_'))

    K.clear_session()
    # config = tf.ConfigProto()
    config = tf.ConfigProto(device_count={'GPU': n_gpu, 'CPU': n_cpu})
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_name = 'cnvnet'
    base_model = cnv_net(n_win_size, n_feat, n_class, filters=32, kernel_size=16, strides=1, pool_size=2,
                         pool_stride=2, drop=drop, blocks=_blocks, fc_size=fc_size,
                         kernel_regular_l2=None, m_name=model_name)
    # base_model = cnv_net_seq(n_win_size, n_feat, n_class)
    if n_gpu > 1:
        model = multi_gpu_model(base_model, n_gpu)
    else:
        model = base_model

    _model_dir = os.path.join(model_root_dir, 'final_model/model_weight')
    if not os.path.isdir(_model_dir):
        os.makedirs(_model_dir)
    _tb_dir = os.path.join(model_root_dir, 'final_model/tb_logs')
    if not os.path.isdir(_tb_dir):
        os.makedirs(_tb_dir)
    _csvlogger_dir = os.path.join(model_root_dir, 'final_model/model_csvlogger')
    if not os.path.isdir(_csvlogger_dir):
        os.makedirs(_csvlogger_dir)

    model_fn = os.path.join(_model_dir, '{}-{}.hdf5'.format(out_name(), model_name))
    tensorboard_fn = os.path.join(_tb_dir, '{}-{}'.format(out_name(), model_name))
    csvlogger_fn = os.path.join(_csvlogger_dir, '{}-{}'.format(out_name(), model_name))
    callbacks = [
        # Early stopping definition
        EarlyStopping(monitor='val_acc', patience=5, verbose=1),
        # Decrease learning rate by 0.5 factor
        AdvancedLearnignRateScheduler(monitor='val_acc', patience=1, verbose=1, mode='auto', decayRatio=0.5),
        # Saving best model
        MultiGPUCheckpointCallback(model_fn, base_model=base_model, monitor='val_acc',
                                   save_best_only=True, verbose=1, save_weights_only=True),
        TensorBoard(tensorboard_fn, batch_size=batch, histogram_freq=2),
        CSVLogger(csvlogger_fn)
    ]

    model.compile(optimizer=keras.optimizers.Adam(lr=learn_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # model.fit(k_x_train, k_y_train, validation_data=(k_x_val, k_y_val),
    #           epochs=epochs, batch_size=batch, callbacks=callbacks)

    model.fit_generator(generator=training_generator,
                    validation_data=validation_data,
                    epochs=5,
                    use_multiprocessing=True,
                    workers=10)