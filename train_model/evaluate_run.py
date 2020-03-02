#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: evaluate_run.py
    Description:
    
Created by Yong Bai on 2019/8/20 3:11 PM.
"""
import os
import numpy as np
import pandas as pd
import h5py
import multiprocessing as mp
import logging
import tensorflow as tf
from keras import backend as K

from model import cnv_net
from train_model import CNVDataGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mp_init(l):
    global lock
    lock = l


def load_feat(feat_fn):
    lock.acquire()
    with np.load(feat_fn) as i_feat_map:
        x = i_feat_map['x']
    lock.release()
    return x


def evaluate(test_sample_fn, test_out_root_dir, model_in_root_dir,
             n_win_size=1000, n_feat=13, n_class=3,
             epochs=64, batch=1024, learn_rate=0.001, drop=0.5,
             fc_size=64, blocks='4_4_3', n_cpu=20, class_weights=None):
    # get model name
    _blocks = tuple(int(x) for x in blocks.split('_'))

    def out_name():
        str_blocks = [str(x) for x in blocks.split('_')]
        str_blk = ''.join(str_blocks)
        if class_weights is not None:
            class_weight_label = ''.join(np.array([class_weights[1], class_weights[2], class_weights[0]]).astype(str))
        else:
            class_weight_label = '111'
        return 'b{0}_e{1}_lr{2:.3f}_dr{3:.1f}_fc{4}_blk{5}_win{6}_cw{7}'.format(batch,
                                                                                epochs,
                                                                                learn_rate,
                                                                                drop,
                                                                                fc_size,
                                                                                str_blk,
                                                                                n_win_size,
                                                                                class_weight_label)

    logger.info('loading model... ')

    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_name = 'cnvnet'
    model = cnv_net(n_win_size, n_feat, n_class, filters=32, kernel_size=16, strides=1, pool_size=2,
                    pool_stride=2, drop=drop, blocks=_blocks, fc_size=fc_size,
                    kernel_regular_l2=None, m_name=model_name)
    model_weight_fn = os.path.join(model_in_root_dir, out_name() + '-' + model_name + '.hdf5')
    if not os.path.exists(model_weight_fn):
        raise FileNotFoundError('model weight file not found. {}'.format(model_weight_fn))
    model.load_weights(model_weight_fn)

    test_out_dir = os.path.join(test_out_root_dir, 'test_out')
    if not os.path.isdir(test_out_dir):
        os.mkdir(test_out_dir)
    test_out_fn = os.path.join(test_out_dir, out_name() + '-offline-test.h5')

    test_sample_df = pd.read_csv(test_sample_fn, sep='\t')

    # slow when using generator
    test_samples_fn_arr = test_sample_df[['f_name']].values
    test_samples_true_arr = test_sample_df['cnv_type_encode'].values

    test_batch_generator = CNVDataGenerator(test_samples_fn_arr, batch,
                                            win_size=n_win_size, n_feat=n_feat, n_classes=n_class,
                                            shuffle=False, pred_gen=True)
    # use predict_generator will produce only batch_size * n outputs
    ypred = model.predict_generator(test_batch_generator,
                                    verbose=1, use_multiprocessing=True,
                                    workers=n_cpu, max_queue_size=64)

    logger.info('writing predicted results into file...')
    with h5py.File(test_out_fn, 'w') as test_out:
        test_out.create_dataset('ypred', data=ypred)
        test_out.create_dataset('ytrue', data=test_samples_true_arr)

    logger.info('Predict Done, result saved at {}'.format(test_out_fn))
