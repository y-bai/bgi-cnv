#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: online_CNV_AE_gen_feat.py
    Description:
    
Created by yong on 2020/1/7 3:05 PM.
"""
import os
import sys
import time
import argparse
import logging
import glob
import numpy as np

import online.GPUtil as GPU
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
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
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

    logger.info('loading training dataset...')
    in_data_fnames = os.path.splitext(data_file)
    out_data_fn = in_data_fnames[0]+'_AE_dimreduce' + in_data_fnames[1]
    if os.path.exists(out_data_fn):
        os.remove(out_data_fn)

    data_h5_r = h5py.File(data_file, 'r')
    data_h5_w = h5py.File(out_data_fn, 'w')
    data_h5_r.copy('pred_meta', data_h5_w, name='pred_meta')

    logger.info('reducing dimensionality ...')
    pred_feat_arr = data_h5_r['pred_feat'][:, :, 0:n_feat]
    dim_reduce_data = encoder.predict(pred_feat_arr, batch_size=batch, verbose=1)

    logger.info('writing results to h5 file...')
    data_h5_w.create_dataset('pred_feat', data=dim_reduce_data, dtype=np.float32,
                             compression="gzip", compression_opts=4)

    data_h5_r.close()
    data_h5_w.close()
    logger.info('<<<finished, Results saved at {}'.format(out_data_fn))


def main(args):

    win_size = args.win_size
    step_size = args.step_size

    sample_id = args.sample_id
    chr_id = args.chr_id
    in_dir = args.in_dir

    min_r = args.min_r

    model_dir = args.model_dir

    # remove previous dim reduce files
    dimreduce_flist = glob.glob(os.path.join(in_dir, '*_chr{0}_*_AE_dimreduce.h5'.format(chr_id)))
    for r_file in dimreduce_flist:
        os.remove(r_file)
        logger.info('removed existing dim reduce file: {}'.format(r_file))

    # whole chr pred h5 files.
    pred_part_fname = 'win{0}_step{1}_r{2:.2f}_chr{3}_seg_a_pred_*'.format(
        win_size, step_size, min_r, chr_id)

    # do not change the values below
    t_lr = 0.001
    t_drop = 0.5
    if win_size >= 4000:
        t_batch = 128
    else:
        t_batch = 512
    model_params = {'n_win_size': win_size,
                    'n_feat': 9,
                    'epochs': 100,
                    'batch': t_batch,
                    'learn_rate': t_lr,
                    'drop': t_drop,
                    'l2r': 1e-4
                    }
    logger.info("AE model parameters: {}".format(model_params))
    pred_h5_fn_list = glob.glob(os.path.join(in_dir, pred_part_fname))

    len_pred_h5_fn_list = len(pred_h5_fn_list)
    assert len_pred_h5_fn_list > 0
    logger.info("The total number of input predictable files: {0}".format(len_pred_h5_fn_list))

    for i, in_pred_h5_fn in enumerate(pred_h5_fn_list):
        in_pred_h5_fn_parts = os.path.splitext(in_pred_h5_fn)
        if in_pred_h5_fn_parts[1] not in ['.h5', '.hdf5']:
            logger.error('invalide input file: {}'.format(in_pred_h5_fn))
            sys.exit(1)

        # set enviornment
        time.sleep(10)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        logger.info('waiting available gpu device...')
        while True:
            gpu_id_lst = GPU.getFirstAvailable(order='random', maxMemory=0.001, maxLoad=0.001, attempts=50,
                                                   interval=60)
            if len(gpu_id_lst) > 0:
                break
        logger.info('processing on device id {}...'.format(gpu_id_lst[0]))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id_lst[0])

        logger.info(">>>processing {0}/{1}, file name: {2}".format(
            i+1, len_pred_h5_fn_list, in_pred_h5_fn))

        ae_gen_feat(model_dir, in_pred_h5_fn, **model_params)

    logger.info("Done, results saved at {}".format(in_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create data sample list ')
    parser.add_argument(
        "-w",
        "--win_size",
        type=int)

    parser.add_argument(
        "-p",
        "--step_size",
        type=int)

    parser.add_argument(
        "-s",
        "--sample_id",
        type=str)

    parser.add_argument(
        "-c",
        "--chr_id",
        type=str)

    parser.add_argument(
        "-i",
        "--in_dir",
        type=str)

    parser.add_argument(
        "-r",
        "--min_r",
        type=float)

    parser.add_argument(
        "-m",
        "--model_dir",
        type=str)


    args = parser.parse_args()
    logger.info('args: {}'.format(args))
    main(args)