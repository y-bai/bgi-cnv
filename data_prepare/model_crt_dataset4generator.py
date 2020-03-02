#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: model_crt_dataset4generator.py
    Description:
    
Created by Yong Bai on 2019/9/17 11:28 AM.
"""
import sys
sys.path.append("..")

import os
import numpy as np
import argparse
import logging

from cnv_utils import str2bool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def cvt_dataset_one_sample(fn, sample_id, out_dataset_dir, m_feats=13, is_norm=False):
    """

    :param fn:
    :param sample_id:
    :param out_dataset_dir:
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
    # npx = np.zeros([n_xs, win_size, m_feats])
    # npy = np.empty(n_xs, dtype="<U10")
    n_neu = 0
    n_del = 0
    n_dup = 0

    for i in range(n_xs):
        i_x = xs[i*m_feats:(i+1) * m_feats, :]

        # normalize
        if not is_norm:
            i_x_max = np.max(i_x, axis=1)
            i_x_max[i_x_max == 0] = 1
            i_x = i_x * 1.0 / i_x_max.reshape(m_feats, 1)
        npx = np.transpose(i_x)
        npy = ys[i].split('|')[4]

        if npy == 'NEU':
            n_neu += 1
            out_fn = os.path.join(out_dataset_dir, sample_id + '_' + npy + '_' + str(n_neu) + '.npz')
        if npy == 'DEL':
            n_del += 1
            out_fn = os.path.join(out_dataset_dir, sample_id + '_' + npy + '_' + str(n_del) + '.npz')
        if npy == 'DUP':
            n_dup += 1
            out_fn = os.path.join(out_dataset_dir, sample_id + '_' + npy + '_' + str(n_dup) + '.npz')
        np.savez_compressed(out_fn, x=npx)


def cvt_dataset_samples(sample_id, in_data_dir, out_dataset_dir,
                        win_size, min_r, min_f, is_norm, m_feats=13):
    """

    :param sample_id:
    :param in_data_dir:
    :param out_dataset_dir:
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
    i_full_fn = os.path.join(in_data_dir, i_fn)

    if not (os.path.exists(i_full_fn + '.x') and os.path.exists(i_full_fn + '.y')):
        raise FileNotFoundError("data file not does not exist for sample {}".format(i_full_fn))

    cvt_dataset_one_sample(i_full_fn, sample_id, out_dataset_dir, m_feats=m_feats, is_norm=is_norm)

    logger.info('Finished sample {0}...'.format(sample_id))


def main(args):

    sample_id = args.sample_id
    win_size = args.win_size
    min_r = args.ratio
    min_f = args.frequency
    is_norm = args.normalize

    out_dir_root = args.out_dir
    in_data_dir = args.in_data_dir

    out_data_folder_name = 'w{0}_r{1:.2f}_f{2:.2f}'.format(win_size, min_r, min_f)

    out_data_subdir = os.path.join(out_dir_root, out_data_folder_name)
    if not os.path.isdir(out_data_subdir):
        os.mkdir(out_data_subdir)

    # final dataset dir
    out_dataset_dir = os.path.join(out_data_subdir, 'dataset')
    if not os.path.isdir(out_dataset_dir):
        os.mkdir(out_dataset_dir)

    cvt_dataset_samples(sample_id, in_data_dir, out_dataset_dir,
                        win_size, min_r, min_f, is_norm, m_feats=13)

    logger.info('Done, data save at {}'.format(out_data_subdir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create data sample list ')
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
        help='if normalizing the training data, if input is .x,.y,.f files, then the data is not normalized')

    parser.add_argument(
        "-i",
        "--in_data_dir",
        type=str,
        help="input dir where storing in text feature files(ie, .x,.y,.f files)")

    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="output directory",
        required=True)

    args = parser.parse_args()
    logger.info('args: {}'.format(args))
    main(args)
