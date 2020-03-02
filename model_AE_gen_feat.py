#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: model_AE_gen_feat.py
    Description:
    
Created by yong on 2020/1/7 3:05 PM.
"""
import os
import argparse
import logging
import numpy as np

import tensorflow as tf
import keras
from keras import backend as K
from keras import Model
from model import cnv_aenet
from train_model import CNVDataGenerator_AE

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ae_gen_feat(model_dir, data_file, n_win_size=10000, n_feat=9,
             epochs=64, batch=128, learn_rate=0.001, drop=0.5, l2r=1e-4):

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
    # load model weight
    model_weight_fn = os.path.join(model_dir, 'model_weight/{}-{}.hdf5'.format(out_name(), model_name))
    logger.info('model weight file: {}'.format(model_weight_fn))
    if not os.path.exists(model_weight_fn):
        raise FileNotFoundError('model weight not exist: {}'.format(model_weight_fn))
    base_model.load_weights(model_weight_fn)

    logger.info('load encoder...')
    encoder = Model(base_model.input, base_model.get_layer('encoder_output').output)
    # print(encoder.summary())

    logger.info('load training dataset...')
    in_data_fnames = os.path.splitext(data_file)
    out_data_fn = in_data_fnames[0]+'_AE_dimreduce' + in_data_fnames[1]
    if os.path.exists(out_data_fn):
        os.remove(out_data_fn)

    data_h5_r = h5py.File(data_file, 'r')
    data_sample_len = data_h5_r['y'].shape[0]
    data_h5_w = h5py.File(out_data_fn, 'w')
    data_h5_r.copy('y', data_h5_w, name='y')
    data_h5_r.copy('original_index', data_h5_w, name='original_index')
    data_h5_r.close()

    logger.info('reduce dim by AE...')
    test_batch_generator = CNVDataGenerator_AE(data_file, data_sample_len, batch,
                                            win_size=n_win_size, n_feat=n_feat,
                                            shuffle=False, pred_gen=True)
    # check data
    # x = test_batch_generator[0]
    # print(x.shape)
    # print(x[0, 0:10, 0:10])
    # print(x[127, 0:10, 0:10])
    # check data end

    x_reduced = encoder.predict_generator(test_batch_generator,
                                    verbose=1, use_multiprocessing=True,
                                    workers=10, max_queue_size=64)

    logger.info('x_reduced.shape = {}'.format(x_reduced.shape))
    data_h5_w.create_dataset('x', data=x_reduced, dtype=np.float32,
                                               chunks=(4, x_reduced.shape[1], x_reduced.shape[2]),
                                               compression="gzip", compression_opts=4)

    data_h5_w.close()
    logger.info('Done, results saved at {}'.format(out_data_fn))


def main(args):
    win_size = args.win_size
    datat = args.datat
    root_dir = "/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out"
    min_r = 0.1
    min_f = 0.01
    train_ratio = 0.8
    nb_cls_prop = '1:2:3'

    if datat == 'train':
        data_filename = 'trains_mr{0:.2f}_mf{1:.2f}_train_{2}_tr{3:.2f}.h5'.format(
            min_r, min_f, ''.join(nb_cls_prop.split(':')), train_ratio)
    else:
        data_filename = 'trains_mr{0:.2f}_mf{1:.2f}_val_{2}_tr{3:.2f}.h5'.format(
            min_r, min_f, ''.join(nb_cls_prop.split(':')), train_ratio)
    # data_filename = 'trains_mr{0:.2f}_mf{1:.2f}_train_111.h5'.format(min_r, min_f)

    data_fname = os.path.join(root_dir, 'data/win_{0}/{1}'.format(win_size, data_filename))
    model_root_dir = os.path.join(root_dir, 'model{}'.format(''.join(nb_cls_prop.split(':'))))

    # do not change the values below
    t_lr = 0.001
    t_drop = 0.5
    if win_size >= 10000:
        t_batch = 128
    elif 4000 <= win_size < 10000:
        t_batch = 128
    else:
        t_batch = 512
        t_lr = 0.001

    logger.info('batch_size: {}'.format(t_batch))
    model_params = {'n_win_size': win_size,
                    'n_feat': 9,
                    'epochs': 100,
                    'batch': t_batch,
                    'learn_rate': t_lr,
                    'drop': t_drop,
                    'l2r': 1e-4
                    }

    logger.info('data file: {}'.format(data_fname))
    ae_gen_feat(model_root_dir, data_fname, **model_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create data sample list ')
    parser.add_argument(
        "-w",
        "--win_size",
        type=int)
    parser.add_argument(
        "-d",
        "--datat",
        type=str)

    args = parser.parse_args()
    logger.info('args: {}'.format(args))
    main(args)