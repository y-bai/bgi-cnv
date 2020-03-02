#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: train_val_dataset.py
    Description:
    
Created by Yong Bai on 2019/9/18 5:14 PM.
"""

import os
import numpy as np
import pandas as pd

import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_train_val_dataset(train_sample_list_fn, out_train_fn, out_val_fn,
                             train_ratio=0.8, nb_cls_prop='1:2:3'):
    """
    create partail training and validation data set, whether the number of data samples for
     each class is equal or not. (ie, do not care imbalance).
     n_del:n_dup:n_neu=1:2:4
    :param train_sample_list_fn: csv files containing all the training data samples,
        resulting from train_test_data,
        naming: 'w{0}_r{1:.2f}_f{2:.2f}_train_all.csv'.format(win_size, min_r, min_f)
        this file has header = ['f_name', 'cnv_type_encode'].
            f_name: the .npz absolute file name containing the feature map with shape (win_size, n_feat).
            cnv_type_encode: encoding int number for class labels: 1: DEL, 2: DUP, 0: NEU
    :param out_train_fn: output csv file for training data samples, format is the same as train_sample_list_fn
    :param out_val_fn: output csv file for validation data samples, format is the same as train_sample_list_fn
    :param train_ratio: the ratio of samples from train_sample_list_fn for training,
        the remaining is for validation
    :param nb_cls_prop: the number of samples between classes. DEL:DUP:NEU
        if balancing data sample, nb_cls_prop = 1,
            and selecting the min number of class-samples to balance samples
        if nb_cls_prop = 1:2:3, we will select the number of DEL as basic-number,
            and sampling the number of samples for other class
    :return:
    """
    logger.info('loading the whole train data sample list file, will take a while...')
    all_train_samples = pd.read_csv(train_sample_list_fn, sep='\t')

    del_samples = all_train_samples[all_train_samples['cnv_type_encode'] == 1]
    dup_samples = all_train_samples[all_train_samples['cnv_type_encode'] == 2]
    neu_samples = all_train_samples[all_train_samples['cnv_type_encode'] == 0]

    del all_train_samples

    logger.info('Calculating the number of samples for each of the class...')
    n_dels = len(del_samples)
    n_dups = len(dup_samples)
    n_neus = len(neu_samples)

    nb_cls_prop_arr = np.array([int(x) for x in nb_cls_prop.split(':')])
    if len(np.unique(nb_cls_prop_arr)) == 1:
        n_sample_arr = np.array([n_dels, n_dups, n_neus])
        n_min = np.min(n_sample_arr)
        n_f_dels = n_min
        n_f_dups = n_min
        n_f_neus = n_min
    else:
        assert n_dups >= n_dels * nb_cls_prop_arr[1]
        assert n_neus >= n_dels * nb_cls_prop_arr[2]

        n_f_dels = n_dels
        n_f_dups = n_dels * nb_cls_prop_arr[1]
        n_f_neus = n_dels * nb_cls_prop_arr[2]
    logger.info('the number to be sampled for DEL, DUP, NEU: {},{},{}'.format(n_f_dels, n_f_dups, n_f_neus))

    logger.info('sampling data sample list...')
    # random sample the samples to make the balance data set
    sel_del_samples = del_samples.sample(n=n_f_dels, random_state=123).reset_index(drop=True)
    sel_dup_samples = dup_samples.sample(n=n_f_dups, random_state=123).reset_index(drop=True)
    sel_neu_samples = neu_samples.sample(n=n_f_neus, random_state=123).reset_index(drop=True)

    logger.info('creating training data set and validation data set for dup, del and neu, respectively...')
    # split to get val data set and train dataset
    del_msk = np.random.rand(n_f_dels) < train_ratio
    train_del_samples = sel_del_samples[del_msk]
    val_del_samples = sel_del_samples[~del_msk]

    dup_msk = np.random.rand(n_f_dups) < train_ratio
    train_dup_samples = sel_dup_samples[dup_msk]
    val_dup_samples = sel_dup_samples[~dup_msk]

    neu_msk = np.random.rand(n_f_neus) < train_ratio
    train_neu_samples = sel_neu_samples[neu_msk]
    val_neu_samples = sel_neu_samples[~neu_msk]

    logger.info('combine training data set of dup, del and neu, and shuffling...')
    train_samples = pd.concat([train_del_samples, train_dup_samples, train_neu_samples], ignore_index=True)
    # shuffle rows
    train_samples = train_samples.sample(frac=1).reset_index(drop=True)

    logger.info('combine validation data set of dup, del and neu, and shuffling...')
    val_samples = pd.concat([val_del_samples, val_dup_samples, val_neu_samples], ignore_index=True)
    # shuffle rows
    val_samples = val_samples.sample(frac=1).reset_index(drop=True)

    logger.info('writing training and validation data set into the file...')
    train_samples.to_csv(out_train_fn, sep='\t', index=False)
    val_samples.to_csv(out_val_fn, sep='\t', index=False)

    logger.info('Done, the result is saved at \n{0}\n{1}'.format(out_train_fn, out_val_fn))


def main(args):

    win_size = args.win_size
    min_r = args.ratio
    min_f = args.frequency
    train_rate = args.train_rate
    in_data_root_dir = args.in_data_root_dir
    out_data_root_dir = args.out_data_root_dir

    nb_cls_prop = args.nb_cls_prop

    train_data_samples_fn = os.path.join(in_data_root_dir,
                                         'w{0}_r{1:.2f}_f{2:.2f}_train_all.csv'.format(win_size, min_r, min_f))
    if not os.path.exists(train_data_samples_fn):
        raise FileNotFoundError('all training data sample list file not found: {}'.format(train_data_samples_fn))

    out_train_fn = os.path.join(out_data_root_dir,
                                'w{0}_r{1:.2f}_f{2:.2f}_train_train_{3}.csv'.format(
                                    win_size, min_r, min_f, ''.join(nb_cls_prop.split(':'))))
    out_val_fn = os.path.join(out_data_root_dir,
                              'w{0}_r{1:.2f}_f{2:.2f}_train_val_{3}.csv'.format(
                                  win_size, min_r, min_f, ''.join(nb_cls_prop.split(':'))))

    if os.path.exists(out_train_fn):
        os.remove(out_train_fn)
    if os.path.exists(out_val_fn):
        os.remove(out_val_fn)

    create_train_val_dataset(train_data_samples_fn,
                             out_train_fn, out_val_fn, train_ratio=train_rate, nb_cls_prop=nb_cls_prop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create training and validation data set ')

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
        "--train_rate",
        type=float,
        default=0.8,
        help="split rate for train samples")

    parser.add_argument(
        "-b",
        "--nb_cls_prop",
        type=str,
        default='1:1:1',
        help="the number of samples between classes. DEL:DUP:NEU")

    parser.add_argument(
        "-i",
        "--in_data_root_dir",
        type=str,
        default='/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data',
        help="input dir for train all data set file: w{0}_r{1:.2f}_f{2:.2f}_train_all.csv")

    parser.add_argument(
        "-o",
        "--out_data_root_dir",
        type=str,
        default='/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data',
        help="out dir for train and validation")

    args = parser.parse_args()
    logger.info('args: {}'.format(args))
    main(args)







