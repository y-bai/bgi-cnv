#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: train_test_data.py
    Description:
    
Created by Yong Bai on 2019/8/20 3:04 PM.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_train_test_ids(data_root_dir, total_samples_ls_fn, test_size=0.1):
    """

    :param data_root_dir: data root dir
      default: /zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data
    :param total_samples_ls_fn: 1KGP sample ID list
      default: /zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/sample.list
    :param test_size:
    :return:
    """

    # create the train and test sample ids
    logger.info("Loading training and testing sample ids...")
    train_test_sample_list = os.path.join(data_root_dir, 'train_test{:.2f}_sample_ids.npz'.format(test_size))
    if not os.path.exists(train_test_sample_list):
        if not os.path.exists(total_samples_ls_fn):
            raise FileNotFoundError('sample list file does not exist. {}'.format(total_samples_ls_fn))
        else:
            sample_id_map = np.loadtxt(total_samples_ls_fn, delimiter='\t', usecols=(0, 1), dtype='str')
            sample_id_arr = sample_id_map[:, 0]

            # remove sample_id = 'NA12878' as we will put this sample in the test dataset
            rm_sample_arr = np.delete(sample_id_arr, np.where(sample_id_arr == 'NA12878'))

            train_ids, tmp_test_ids = train_test_split(rm_sample_arr, test_size=test_size, random_state=123)
            test_ids = np.append(tmp_test_ids, 'NA12878')
            np.savez(train_test_sample_list, train_ids=train_ids, test_ids=test_ids)
            return train_ids, test_ids
    else:
        with np.load(train_test_sample_list) as sample_ids:
            return sample_ids['train_ids'], sample_ids['test_ids']


def data_sample_stat(fsn_pd, sample_ids_arr=None):
    """

    :param fsn_pd:
    :param sample_ids_arr:
    :return:
    """
    fsn_pd['sample_id'] = fsn_pd['f_name'].str.split('/').str[-1].str.split('_').str[0]
    fsn_pd['cnv_type'] = fsn_pd['f_name'].str.split('/').str[-1].str.split('_').str[1]
    fsn_pd['cnv_type_encode'] = fsn_pd['cnv_type'].apply(lambda x: 1 if x == 'DEL' else 2 if x == 'DUP' else 0)

    re_fsn_pd = fsn_pd[fsn_pd['sample_id'].isin(sample_ids_arr)] if sample_ids_arr is not None else fsn_pd

    return re_fsn_pd[['f_name', 'cnv_type_encode']]


def data_samples_train_test(data_root_dir, total_samples_ls_fn, win_size, min_f, min_r, test_size=0.1):
    """

    :param data_root_dir:
    :param total_samples_ls_fn:
    :param win_size:
    :param min_f:
    :param min_r:
    :param test_size:
    :return:
    """

    # get train and test sample ids
    train_sample_ids, test_sample_ids = get_train_test_ids(data_root_dir, total_samples_ls_fn, test_size=test_size)

    # get data sample file list for the given win_size, min_f and min_r
    # the name format of w{0}_r{1:.2f}_f{2:.2f}_feat_list.txt is created by run10.sh script
    data_sample_fn_list = os.path.join(data_root_dir,
                                       'w{0}_r{1:.2f}_f{2:.2f}_feat_list.txt'.format(win_size, min_r, min_f))

    train_data_samples_fn = os.path.join(data_root_dir,
                                         'w{0}_r{1:.2f}_f{2:.2f}_train_all.csv'.format(win_size, min_r, min_f))
    test_data_samples_fn = os.path.join(data_root_dir,
                                        'w{0}_r{1:.2f}_f{2:.2f}_test.csv'.format(win_size, min_r, min_f))

    logger.info("creating data sample list for train and test set, respectively...")
    if os.path.exists(train_data_samples_fn):
        os.remove(train_data_samples_fn)
    if os.path.exists(test_data_samples_fn):
        os.remove(test_data_samples_fn)

    for fns_pd in pd.read_csv(data_sample_fn_list,
                              header=None,
                              names=['f_name'],
                              chunksize=200000):
        train_data_samples = data_sample_stat(fns_pd, sample_ids_arr=train_sample_ids)
        test_data_samples = data_sample_stat(fns_pd, sample_ids_arr=test_sample_ids)

        if os.path.exists(train_data_samples_fn):
            train_data_samples.to_csv(train_data_samples_fn, sep='\t', mode='a', header=False, index=False)
        else:
            train_data_samples.to_csv(train_data_samples_fn, sep='\t', mode='a', header=True, index=False)

        if os.path.exists(test_data_samples_fn):
            test_data_samples.to_csv(test_data_samples_fn, sep='\t', mode='a', header=False, index=False)
        else:
            test_data_samples.to_csv(test_data_samples_fn, sep='\t', mode='a', header=True, index=False)

    logger.info("Done, save the data at {0} and {1}".format(train_data_samples_fn, test_data_samples_fn))


def main(args):

    win_size = args.win_size
    min_r = args.ratio
    min_f = args.frequency
    in_data_root_dir = args.in_data_root_dir
    sample_id_list_fn = args.sample_id_list_fn
    test_sample_rate = args.test_sample_rate

    data_samples_train_test(in_data_root_dir, sample_id_list_fn, win_size, min_f, min_r, test_size=test_sample_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create training and testing sample list ')

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
        "-t",
        "--test_sample_rate",
        type=float,
        default=0.1,
        help="split rate for test samples")

    parser.add_argument(
        "-i",
        "--in_data_root_dir",
        type=str,
        default='/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data',
        help="input dir all dada sample list file: w{0}_r{1:.2f}_f{2:.2f}_feat_list.txt")

    parser.add_argument(
        "-l",
        "--sample_id_list_fn",
        type=str,
        help="1kgp sample list file name",
        default='/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/sample.list')

    args = parser.parse_args()
    logger.info('args: {}'.format(args))
    main(args)
