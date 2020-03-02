#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: mat_crt_train_data.py
    Description: create training data set

Created by Yong Bai on 2019/7/26 1:06 PM.
"""
import sys
sys.path.append("..")
import argparse
import logging
import os
import numpy as np

import pandas as pd
from io import StringIO
from cnv_utils import str2bool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_feat_block(seq, block_by):
    """
    A generator that splits a read cnv feature file block by block that is indicated by #
    :param seq: input cnv features
    :param block_by: indicator of the block, which is #
    :return: iterator of block data
    """
    dat = []
    for line in seq:
        if line.startswith(block_by):
            if dat:
                yield dat
                dat = []

        dat.append(line)

    if dat:
        yield dat


def feat_cnvt(dat_blok, gaps, win_size, min_r, min_f, is_norm, input_feature_type):
    """

    :param dat_blok:
    :param gaps:
    :param win_size:
    :param min_r:
    :param min_f:
    :param is_norm:
    :param input_feature_type:
    :return:
    """
    if input_feature_type == 1:
        len_dat_block = 15
    else:
        len_dat_block = 14
    assert len(dat_blok) == len_dat_block  # if DEL or DUP then , 15

    blk_heads = dat_blok[0].split('[')
    # logger.info('block head: {}'.format(dat_blok[0]))
    cnv_infos = blk_heads[0][1:-1].split(',')  # ignore the latest ','
    reads_relative_poss = blk_heads[1][:-2]  # ignore the latest ']\n'

    assert len(cnv_infos) == 6

    # we do not consider chr X, Y
    chr_id = cnv_infos[0]
    if chr_id == 'X':
        return False, cnv_infos, 'X'

    cnv_abs_start = int(cnv_infos[1])
    cnv_abs_end = int(cnv_infos[2])
    cnv_len = int(cnv_infos[3])
    cnv_type = cnv_infos[4]
    cnv_freq = float(cnv_infos[5])

    if cnv_freq < min_f:
        logger.info('CNV frequncy(={}) does not meet min value(={}).'.format(cnv_freq, min_f))
        return False, cnv_infos, 'FREQ_LOW'
    if cnv_len < win_size:
        logger.info('CNV len(={}) does not meet min value(={}).'.format(cnv_len, win_size))
        return False, cnv_infos, 'LEN_LOW'

    i_gaps = gaps.loc[(gaps['CHROM'] == 'chr' + chr_id) &
                      (gaps['START'] >= cnv_abs_start) &
                      (gaps['START'] <= cnv_abs_end), ['START', 'END']]
    logger.info('gap size = {}'.format(i_gaps.shape))

    gaps_poss = []
    for _, igrow in i_gaps.iterrows():
        itmp = list(range(igrow['START'] - cnv_abs_start, igrow['END'] + 1 - cnv_abs_start))
        gaps_poss = gaps_poss + itmp
    # if len(gaps_poss) > 0:
    #     logger.info('gap position: {}'.format(gaps_poss))
    # else:
    #     logger.info('No gap in the cnv region!')

    # check number of coverage
    if len(reads_relative_poss) > 0:
        reads_rel_pos_arr = [int(x) for x in reads_relative_poss.split(',')]
        remain_cov_pos = list(set(reads_rel_pos_arr) - set(gaps_poss))
    else:
        remain_cov_pos = []

    # wo have to consider base quality and base map quality
    n_cov_pos = len(remain_cov_pos)

    final_mat = []
    labels = []
    if n_cov_pos < win_size * min_r:
        logger.info('Not enough coverage position for the cnv region')
        return False, cnv_infos, 'COV_LOW(N={})'.format(n_cov_pos)

    # only use 13 features
    f_mat = np.loadtxt(StringIO(''.join(dat_blok[1:14])))
    m, n = f_mat.shape
    if n < win_size:  # this code won't be reached
        # padding zeros
        f_tmp_mat = np.pad(f_mat, ((0, 0), (0, win_size - n)), 'constant')
        if is_norm:
            f_tmp_mat = f_tmp_mat * 1.0 / np.max(f_tmp_mat, axis=1)
        final_mat.append(f_tmp_mat)
        labels.append('|'.join(cnv_infos + ['0']))
        logger.info('after padding,f_mat shape: '.format(f_mat.shape))
    else:
        # slide window
        i_slides = n // win_size
        i_remain = n % win_size

        for i_s in range(1, i_slides + 1):
            f_mat_idx = range((i_s - 1) * win_size, i_s * win_size)
            i_n_cov = set(remain_cov_pos).intersection(set(f_mat_idx))

            if len(i_n_cov) >= win_size * min_r:
                f_tmp_mat = f_mat[:, f_mat_idx]
                if is_norm:
                    f_mat_max = np.max(f_tmp_mat, axis=1)
                    f_mat_max[f_mat_max == 0] = 1
                    f_tmp_mat = f_tmp_mat * 1.0 / f_mat_max.reshape(m, 1)

                final_mat.append(f_tmp_mat)
                labels.append('|'.join(cnv_infos + [str(i_s - 1)]))
        if i_remain > 0:
            # add this part
            f_mat_idx = range(n - win_size, n)
            i_n_cov = set(remain_cov_pos).intersection(set(f_mat_idx))
            if len(i_n_cov) >= win_size * min_r:
                f_tmp_mat = f_mat[:, n - win_size:n]
                if is_norm:
                    f_mat_max = np.max(f_tmp_mat, axis=1)
                    f_mat_max[f_mat_max == 0] = 1
                    f_tmp_mat = f_tmp_mat * 1.0 / f_mat_max.reshape(m, 1)
                final_mat.append(f_tmp_mat)
                labels.append('|'.join(cnv_infos + [str(i_s)]))
    return True, final_mat, labels


def main(args):
    """
    main function to create train data sets
    :param args: stdin args
    :return: no returns
    """

    sample_id = args.sample_id
    feature_fn = args.feature_fn
    win_size = args.win_size
    out_dir = args.out_dir
    min_r = args.ratio
    min_f = args.frequency
    is_norm = args.normalize
    input_feature_type = args.in_feat_type

    if not os.path.exists(feature_fn):
        raise FileNotFoundError('{} does not exist.'.format(feature_fn))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    logger.info(
        'Creating training set for sample {}, window size={}'.format(
            sample_id, win_size))

    # load reference genome gaps
    # header
    # gap position starts with 0 index
    if not os.path.exists('../hg38_config/gap.txt'):
        raise FileNotFoundError('../hg38_config/gap.txt not found.')
    col_names = ['BIN', 'CHROM', 'START', 'END', 'IX', 'N', 'SIZE', 'TYPE', 'BRIDGE']
    col_dtype = [int, str, int, int, int, str, int, str, str]
    gap_header = dict(zip(col_names, col_dtype))
    gap_pd = pd.read_csv('../hg38_config/gap.txt', sep='\t', header=None, names=col_names, dtype=gap_header)

    # load feature

    out_f_title = 'ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.{0}.cnvs.training.{1}_{2:.2f}_{3:.2f}_{4}' \
        .format(sample_id, win_size, min_r, min_f, is_norm)

    out_f_x = os.path.join(out_dir, out_f_title + '.x')
    out_f_y = os.path.join(out_dir, out_f_title + '.y')
    out_f_f = os.path.join(out_dir, out_f_title + '.f')

    if os.path.exists(out_f_x):
        os.remove(out_f_x)
    if os.path.exists(out_f_y):
        os.remove(out_f_y)
    if os.path.exists(out_f_f):
        os.remove(out_f_f)

    with open(feature_fn, 'r') as fin, open(out_f_x, 'a') as ofx, open(out_f_y, 'a') as ofy, open(out_f_f, 'a') as off:
        for i_blocks in read_feat_block(fin, '#'):
            i_block_re = feat_cnvt(i_blocks, gap_pd, win_size, min_r, min_f, is_norm, input_feature_type)
            if i_block_re[0]:
                # write to result files
                # logger.info('i_block_re[1]: {}'.format(len(i_block_re[1])))
                # logger.info(i_block_re[2])

                for i_f in range(len(i_block_re[1])):
                    np.savetxt(ofx, i_block_re[1][i_f], fmt='%-10.5f')
                    ofy.write('{}\n'.format(i_block_re[2][i_f]))
            else:
                # write to filter out files
                off.write('{}|{}\n'.format('|'.join(i_block_re[1]), i_block_re[2]))
    logger.info('Done, sample: {}'.format(sample_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create training dataset')
    parser.add_argument(
        "-s",
        "--sample_id",
        type=str,
        help="sample id",
        required=True)

    parser.add_argument(
        "-f",
        "--feature_fn",
        type=str,
        help="cnv features for the given sample",
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
        default=True,
        help='normalize the training data')

    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="output directory",
        required=True)

    parser.add_argument(
        "-t",
        "--in_feat_type",
        type=int,
        default=1,
        help="input feature type. 1: cnv feature, 0: neu feature",
        required=True)

    args = parser.parse_args()
    main(args)
