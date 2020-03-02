#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: cv_train.py
    Description: cross validtion
    
Created by Yong Bai on 2019/8/26 11:31 AM.
"""

import os
import gc
import numpy as np
from model import cnv_net

import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger
from cnv_utils import AdvancedLearnignRateScheduler, MultiGPUCheckpointCallback
from keras.utils import multi_gpu_model
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args

import horovod.keras as hvd

import pickle

best_accuracy = 0.0


def mnd_train(x_train, y_train, model_root_dir, n_gpu=4, n_cpu=10):
    """
    multiple node distributed train
    :param x_train:
    :param y_train:
    :param model_root_dir:
    :param n_gpu:
    :param n_cpu:
    :return:
    """

    # Horovod: initialize Horovod
    hvd.init()

    K.clear_session()
    gc.collect()
    # config = tf.ConfigProto(device_count={'GPU': n_gpu, 'CPU': n_cpu})
    # Horovod: pin GPU to be used to process local rank(one GPU perprocess)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)
    K.set_session(sess)

    n_samples, n_win_size, n_feat = x_train.shape
    n_class = y_train.shape[1]

    # search-dimension
    # dim_nb_batchs = Categorical(categories=[128, 256, 512, 1024], name='batch_size')
    # dim_nb_epochs = Categorical(categories=[20, 30, 40, 50], name='epoch')
    # dim_lrs = Categorical(categories=[0.1, 0.01, 0.001], name='learn_rate')
    # dim_lr_decays = Categorical(categories=[0.1, 0.5, 0.8], name='learn_rate_decay')
    # dim_init_filters = Categorical(categories=[16, 32, 64, 128], name='filters')
    # dim_drops = Categorical(categories=[0.2, 0.3, 0.4, 0.5], name='drop')
    # dim_fc_sizes = Categorical(categories=[32, 64, 128, 256], name='fc_size')
    # dim_net_blocks = Categorical(categories=[(4, 1), (4, 3), (4, 4, 1), (4, 4, 3), (4, 4, 4, 1), (4, 4, 4, 3)],
    #                              name='blocks')
    # search_dim = [dim_nb_batchs,
    #               dim_nb_epochs,
    #               dim_lrs,
    #               dim_lr_decays,
    #               dim_init_filters,
    #               dim_drops,
    #               dim_fc_sizes,
    #               dim_net_blocks]
    # default_param = [256, 20, 0.1, 0.8, 16, 0.2, 64, (4, 1)]

    dim_nb_batchs = Categorical(categories=[256], name='batch_size')
    dim_nb_epochs = Categorical(categories=[5], name='epoch')
    dim_lrs = Categorical(categories=[0.1, 0.01], name='learn_rate')
    dim_lr_decays = Categorical(categories=[0.8], name='learn_rate_decay')
    dim_init_filters = Categorical(categories=[128], name='filters')
    dim_drops = Categorical(categories=[0.5], name='drop')
    dim_fc_sizes = Categorical(categories=[256], name='fc_size')
    dim_net_blocks = Categorical(categories=[(4, 1)],
                                 name='blocks')
    search_dim = [dim_nb_batchs,
                  dim_nb_epochs,
                  dim_lrs,
                  dim_lr_decays,
                  dim_init_filters,
                  dim_drops,
                  dim_fc_sizes,
                  dim_net_blocks]
    default_param = [256, 5, 0.1, 0.8, 16, 0.2, 64, (4, 1)]

    _model_dir = os.path.join(model_root_dir, 'models3/model_weight')
    if not os.path.isdir(_model_dir):
        os.mkdir(_model_dir)
    _tb_dir = os.path.join(model_root_dir, 'models3/logs')
    if not os.path.isdir(_tb_dir):
        os.mkdir(_tb_dir)
    _csvlogger_dir = os.path.join(model_root_dir, 'models3/model_metrics')
    if not os.path.isdir(_csvlogger_dir):
        os.mkdir(_csvlogger_dir)

    def out_name(batch_size, epoch, learn_rate, learn_rate_decay, filters, drop, fc_size, blocks):
        str_blocks = [str(x) for x in blocks]
        str_blk = ''.join(str_blocks)

        return 'b{0}_e{1}_lr{2:.3f}_lrd{3:.1f}_flt{4}_dr{5:.1f}_fc{6}_blk{7}'.format(batch_size,
                                                                                     epoch,
                                                                                     learn_rate,
                                                                                     learn_rate_decay,
                                                                                     filters,
                                                                                     drop,
                                                                                     fc_size,
                                                                                     str_blk)

    # y_train_labels = np.argmax(y_train, axis=1)
    # skf = StratifiedKFold(n_splits=1, random_state=123, shuffle=True)

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    permutation = list(np.random.permutation(n_samples))

    x_train = x_train[permutation]
    y_train = y_train[permutation]

    @use_named_args(dimensions=search_dim)
    def gp_fitness(batch_size, epoch, learn_rate, learn_rate_decay, filters, drop, fc_size, blocks):
        print('batch_size: {}'.format(batch_size))
        print('epoch: {}'.format(epoch))
        print('learn rate: {0:.3f}'.format(learn_rate))
        print('learn rate decay: {0:.1f}'.format(learn_rate_decay))
        print('filters: {}'.format(filters))
        print('drop ratio: {0:.1f}'.format(drop))
        print('fc size: {}'.format(fc_size))
        print('blocks: {}'.format(blocks))

        tmp_out_name = out_name(batch_size, epoch, learn_rate,
                                learn_rate_decay, filters, drop, fc_size, blocks)

        val_acc_arr = []

        # for i, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train_labels)):
        # ix_train1, ix_val1 = x_train[train_idx], x_train[val_idx]
        # iy_train1, iy_val1 = y_train[train_idx], y_train[val_idx]
        for i in range(1):
            ix_train1, ix_val1, iy_train1, iy_val1 = train_test_split(x_train, y_train, test_size=0.2,
                                                                      shuffle=False)
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

            model = cnv_net(n_win_size, n_feat, n_class,
                                 filters=filters, kernel_size=16, strides=1, pool_size=2,
                                 pool_stride=2, drop=drop, blocks=blocks, fc_size=fc_size, m_name=tmp_out_name)

            callbacks = [
                # Horovod: broadcast initial variable states from rank 0 to all other processes.
                # This is necessary to ensure consistent initialization of all workers when
                # training is started with random weights or restored from a checkpoint.
                hvd.callbacks.BroadcastGlobalVariablesCallback(0),

                # # Horovod: average metrics among workers at the end of every epoch.
                # #
                # # Note: This callback must be in the list before the ReduceLROnPlateau,
                # # TensorBoard, or other metrics-based callbacks.
                # hvd.callbacks.MetricAverageCallback(),
                #
                # # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
                # # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
                # # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
                # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=verbose),
                #
                # # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
                # hvd.callbacks.LearningRateScheduleCallback(start_epoch=5, end_epoch=30, multiplier=1.),
                # hvd.callbacks.LearningRateScheduleCallback(start_epoch=30, end_epoch=60, multiplier=1e-1),
                # hvd.callbacks.LearningRateScheduleCallback(start_epoch=60, end_epoch=80, multiplier=1e-2),
                # hvd.callbacks.LearningRateScheduleCallback(start_epoch=80, multiplier=1e-3),
            ]

            # Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
            if hvd.rank() == 0:
                callbacks.append(EarlyStopping(monitor='val_acc', patience=5, verbose=1))
                callbacks.append(AdvancedLearnignRateScheduler(monitor='val_acc', patience=1, verbose=1, mode='auto',
                                              decayRatio=learn_rate_decay))
                callbacks.append(MultiGPUCheckpointCallback(model_fn, base_model=model, monitor='val_acc',
                                           save_best_only=True, verbose=1, save_weights_only=True))
                callbacks.append(TensorBoard(tensorboard_fn, batch_size=batch_size, histogram_freq=2))
                callbacks.append(CSVLogger(csvlogger_fn))

            # Horovod: adjust learning rate based on number of GPUs.
            opt = keras.optimizers.Adam(lr=learn_rate)
            # Horovod: add Horovod Distributed Optimizer.
            opt = hvd.DistributedOptimizer(opt)

            model.compile(optimizer=opt,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            hist = model.fit(k_x_train, k_y_train, validation_data=(k_x_val, k_y_val), verbose=verbose,
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
            i_config.gpu_options.visible_device_list = str(hvd.local_rank())
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
                                n_calls=40,
                                x0=default_param)

    with open(os.path.join(model_root_dir, 'models3/gp_search_res.pickle'), 'wb') as f:
        pickle.dump(search_result, f)
