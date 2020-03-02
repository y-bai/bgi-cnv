#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: model_train_main3.py
    Description:
    
Created by Yong Bai on 2019/8/20 3:13 PM.
"""

import os
from train_model import train3
import multiprocessing as mp
import argparse
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(args):
    win_size = args.win_size

    out_root_dir = "/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out"
    in_data_root_dir = '/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data'

    min_r = 0.1
    min_f = 0.01
    train_ratio = 0.8
    # nb_cls_prop = '1:1:1'
    nb_cls_prop = '1:2:3'

    # Note: class_weight in fit_generator would course OOM
    # https://github.com/tensorflow/tensorflow/issues/31253
    # class_weights = None

    in_data_dir = os.path.join(in_data_root_dir, 'win_{}'.format(win_size))
    # in_train_fn = os.path.join(in_data_dir,
    #                            'trains_mr{0:.2f}_mf{1:.2f}_train_{2}.h5'.format(
    #                                min_r, min_f, ''.join(nb_cls_prop.split(':'))))
    #
    # in_val_fn = os.path.join(in_data_dir,
    #                          'trains_mr{0:.2f}_mf{1:.2f}_val_{2}.h5'.format(
    #                              min_r, min_f, ''.join(nb_cls_prop.split(':'))))

    in_train_fn = os.path.join(in_data_dir,
                               'trains_mr{0:.2f}_mf{1:.2f}_train_{2}_tr{3:.2f}_AE_dimreduce.h5'.format(
                                   min_r, min_f, ''.join(nb_cls_prop.split(':')), train_ratio))

    in_val_fn = os.path.join(in_data_dir,
                             'trains_mr{0:.2f}_mf{1:.2f}_val_{2}_tr{3:.2f}_AE_dimreduce.h5'.format(
                                 min_r, min_f, ''.join(nb_cls_prop.split(':')), train_ratio))

    out_model_root_dir = os.path.join(out_root_dir, 'model{}'.format(''.join(nb_cls_prop.split(':'))))

    if not os.path.exists(in_train_fn):
        raise FileNotFoundError('train sample file not found: {}'.format(in_train_fn))
    if not os.path.exists(in_val_fn):
        raise FileNotFoundError('validation sample file not found: {}'.format(in_val_fn))
    if not os.path.isdir(out_model_root_dir):
        os.mkdir(out_model_root_dir)

    t_lr = 0.001  #
    t_drop = 0.5
    t_cpu = 10
    t_batch = 128

    n_res_blks = -1
    t_win_size = win_size
    while t_win_size % 2 == 0:
        n_res_blks += 1
        t_win_size = int(t_win_size // 2)

    if n_res_blks > 0:
        f_win_size = win_size // (2 ** (n_res_blks + 1))
    else:
        f_win_size = win_size

    logger.info('The win_size is {} after AE.'.format(f_win_size))
    model_params = {'n_win_size': f_win_size,
                    'n_feat': 9,
                    'n_class': 3,
                    'epochs': 100,
                    'batch': t_batch,
                    'learn_rate': t_lr,
                    'drop': t_drop,
                    'fc_size': 64,
                    'blocks': '4_3',
                    'l2r': 1e-4,
                    'temperature': 6,
                    'lbl_smt_frac': 0,
                    'filters': 32,
                    'kernel_size': 16,
                    'n_gpu': 4,
                    'n_cpu': t_cpu,
                    'pw': -8}
    logger.info('params: {}'.format(model_params))

    train3(in_train_fn, in_val_fn, out_model_root_dir, **model_params)


if __name__ == "__main__":
    mp.set_start_method('forkserver')
    # mp.set_start_method('spawn')
    # ref:
    # https://github.com/keras-team/keras/issues/10340
    # https://github.com/keras-team/keras/issues/11671
    # https://github.com/keras-team/keras/issues/10948
    # https://docs.python.org/3.4/library/multiprocessing.html?highlight=process#multiprocessing.set_start_method
    # mp.set_start_method('forkserver')  # thread-safe

    # path for save the model
    parser = argparse.ArgumentParser(description='Create data sample list ')
    parser.add_argument(
        "-w",
        "--win_size",
        type=int)

    args = parser.parse_args()
    logger.info('args: {}'.format(args))
    main(args)






