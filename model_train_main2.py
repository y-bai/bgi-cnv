#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: model_train_main.py
    Description:
    
Created by Yong Bai on 2019/8/20 3:13 PM.
"""

import os
from train_model import train2
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
    nb_cls_prop = '1:1:1'
    # nb_cls_prop = '1:2:3'

    # Note: class_weight in fit_generator would course OOM
    # https://github.com/tensorflow/tensorflow/issues/31253
    # class_weights = None

    in_data_dir = os.path.join(in_data_root_dir, 'win_{}'.format(win_size))
    in_train_fn = os.path.join(in_data_dir,
                               'trains_mr{0:.2f}_mf{1:.2f}_train_{2}.h5'.format(
                                   min_r, min_f, ''.join(nb_cls_prop.split(':'))))

    in_val_fn = os.path.join(in_data_dir,
                             'trains_mr{0:.2f}_mf{1:.2f}_val_{2}.h5'.format(
                                 min_r, min_f, ''.join(nb_cls_prop.split(':'))))

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
    if win_size >= 10000:
        t_batch = 64
    elif 4000 <= win_size < 10000:
        t_batch = 64
    else:
        t_batch = 512
        t_lr = 0.001
        t_cpu = 24

    print('batch_size: {}'.format(t_batch))
    model_params = {'n_win_size': win_size,
                    'n_feat': 13,
                    'n_class': 3,
                    'epochs': 100,
                    'batch': t_batch,  # 512->128
                    'learn_rate': t_lr,  # 0.001->0.01
                    'drop': t_drop,  # 0.5->0.2
                    'fc_size': 64,
                    'blocks': '4_4_3',
                    'l2r': 1e-4,
                    'temperature': 6,  # 4->6
                    'lbl_smt_frac': 0,
                    'filters': 32,
                    'kernel_size': 16,
                    'n_gpu': 4,
                    'n_cpu': t_cpu,  # 20->16
                    'pw': -8}  # opt pw is -8

    train2(in_train_fn, in_val_fn, out_model_root_dir, **model_params)


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






