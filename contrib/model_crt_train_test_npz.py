#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: model_crt_train_test_npz.py
    Description:
    
Created by Yong Bai on 2019/8/5 10:48 AM.
"""
import sys
sys.path.append("..")

import os
import numpy as np

import argparse
import logging
import gc

from cnv_utils import str2bool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def cvt_dataset_one_sample(fn, win_size=1000, m_feats=13, is_norm=False):
    """

    :param fn:
    :param win_size: window size
    :param m_feats:
    :param is_norm:
    :return:
    """

    x_fn = fn + '.x'
    y_fn = fn + '.y'

    xs = np.loadtxt(x_fn)
    with open(y_fn, 'r') as y_f:
        ys = y_f.readlines()

    # 13 features
    n_xs = int(len(xs) / m_feats)
    assert n_xs == len(ys)

    logger.info('Number of examples: {}'.format(n_xs))

    npx = np.zeros([n_xs, win_size, m_feats])
    npy = np.empty(n_xs, dtype="<U10")

    for i in range(n_xs):
        i_x = xs[i*m_feats:(i+1) * m_feats, :]

        # normalize
        if not is_norm:
            i_x_max = np.max(i_x, axis=1)
            i_x_max[i_x_max == 0] = 1
            i_x = i_x * 1.0 / i_x_max.reshape(m_feats, 1)

        npx[i, :, :] = np.transpose(i_x)
        npy[i] = ys[i].split('|')[4]

    return npx, npy


def cvt_dataset_samples(sample_id, data_dir, out_fn, win_size, min_r, min_f, is_norm, m_feats=13):
    """

    :param sample_id:
    :param data_dir:
    :param out_fn:
    :param win_size:
    :param min_r:
    :param min_f:
    :param is_norm:
    :param m_feats:
    :return:
    """

    logger.info('Processing sample {0}...'.format(sample_id))

    i_fn = 'ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.{0}.cnvs.training.{1}_{2:.2f}_{3:.2f}_{4}' \
            .format(sample_id, win_size, min_r, min_f, is_norm)
    i_full_fn = os.path.join(data_dir, i_fn)

    if not os.path.exists(i_full_fn + '.x') or not os.path.exists(i_full_fn + '.y'):
        raise FileNotFoundError("data file not does not exist for sample {}".format(sample_id))

    npx, npy = cvt_dataset_one_sample(i_full_fn, win_size, m_feats, is_norm)
    np.savez_compressed(out_fn, x=npx, y=npy)
    del npx, npy
    gc.collect()
    logger.info('Finished sample {0}...'.format(sample_id))


def create_train_test_npz(data4train_dir, data_dir, sample_id, win_size, min_r, min_f, is_norm):
    """

    :param data4train_dir:
    :param data_dir:
    :param sample_id:
    :param win_size:
    :param min_r:
    :param min_f:
    :param is_norm:
    :return:
    """
    logger.info("Creating data training set and save into {}".format(data4train_dir))

    data4train_fn = os.path.join(data4train_dir,
                                 "ds_data4model_{0}_{1}_{2:.2f}_{3:.2f}_{4}.npz".format(
                                     sample_id, win_size, min_r, min_f, is_norm))

    cvt_dataset_samples(sample_id, data_dir, data4train_fn, win_size, min_r, min_f, is_norm, 13)


def main(args):

    sample_id = args.sample_id

    win_size = args.win_size
    min_r = args.ratio
    min_f = args.frequency
    is_norm = args.normalize

    out_dir_root = args.out_dir
    in_data_dir = args.in_data_dir

    model_save_dir = os.path.join(out_dir_root, 'models')
    data4train_dir = os.path.join(out_dir_root, 'data4models')
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)
    if not os.path.isdir(data4train_dir):
        os.mkdir(data4train_dir)

    create_train_test_npz(data4train_dir, in_data_dir, sample_id, win_size, min_r, min_f, is_norm)

    logger.info('Done, data save at {}'.format(data4train_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create training and testing npz dataset')
    parser.add_argument(
        "-s",
        "--sample_id",
        type=str,
        help="sample id",
        required=True)

    parser.add_argument(
        "-w",
        "--win_size",
        type=int,
        default=1000,
        help='window size. This will be hyperparamter')

    parser.add_argument(
        "-r",
        "--ratio",
        type=float,
        default=0.1,
        help='cnv region that has read coverage less than the ratio will be filtered out. This will be hyperparameter')

    parser.add_argument(
        "-q",
        "--frequency",
        type=float,
        default=0.01,
        help='cnv whose frequency less than the frequency will be filtered out. This will be hyperparameter')

    parser.add_argument(
        "-n",
        "--normalize",
        type=str2bool,
        default=False,
        help='normalize the training data')

    parser.add_argument(
        "-i",
        "--in_data_dir",
        type=str,
        help="input data directory",
        required=True)

    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="output directory",
        required=True)

    args = parser.parse_args()
    logger.info('args: {}'.format(args))
    main(args)


