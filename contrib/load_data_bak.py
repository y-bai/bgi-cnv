#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: train_test_data.py
    Description:
    
Created by Yong Bai on 2019/8/20 3:04 PM.
"""
import os
import gc
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_data(sample_id, model_root_dir, win_size=1000, min_r=0.1, min_f1=0.01, min_f2=-2, is_norm=False):
    """

    :param sample_id:
    :param model_root_dir:
    :param win_size: window size
    :param min_r: minimum value of ratio of base coverage along thr window
    :param min_f1: minimum frequency for DEL or DUP data, default 0.01
    :param min_f2: minimum frequency for NEU data. The value is just only NEU
    :param is_norm: this is the same as the value when creating training dataset in mat_crt_train_data
    :return:
    """
    logger.info('Loading dataset for sample {}...'.format(sample_id))
    data_dir = os.path.join(model_root_dir, 'data4models')
    del_dup_data_fn = os.path.join(data_dir, "ds_data4model_{0}_{1}_{2:.2f}_{3:.2f}_{4}.npz".format(
        sample_id, win_size, min_r, min_f1, is_norm))
    neu_data_fn = os.path.join(data_dir, "ds_data4model_{0}_{1}_{2:.2f}_{3:.2f}_{4}.npz".format(
        sample_id, win_size, min_r, min_f2, is_norm))

    if os.path.exists(del_dup_data_fn) and os.path.exists(neu_data_fn):
        with np.load(del_dup_data_fn) as del_dup_data, np.load(neu_data_fn) as neu_data:
            x = np.concatenate((del_dup_data['x'], neu_data['x']), axis=0)
            y = np.concatenate((del_dup_data['y'], neu_data['y']))
        return x, y

    else:
        logger.warning('sample {} does not have enough data.'.format(sample_id))
        return None, None


def get_train_test_ids(model_root_dir, total_samples_ls_fn, test_size=0.1):
    """

    :param model_root_dir:
    :param total_samples_ls_fn:
    :param test_size:
    :return:
    """

    # create the train and test sample ids
    logger.info("Loading training and testing sample ids...")
    train_test_sample_list = os.path.join(model_root_dir, '1k_train_test{:.2f}_sample_ids.npz'.format(test_size))
    if not os.path.exists(train_test_sample_list):
        if not os.path.exists(total_samples_ls_fn):
            raise FileNotFoundError('sample list file does not exist. {}'.format(total_samples_ls_fn))
        else:
            sample_id_map = np.loadtxt(total_samples_ls_fn, delimiter='\t', usecols=(0, 1), dtype='str')
            sample_id_arr = sample_id_map[:, 0]
            train_ids, test_ids = train_test_split(sample_id_arr, test_size=test_size, random_state=123)
            np.savez(train_test_sample_list, train_ids=train_ids, test_ids=test_ids)
            return train_ids, test_ids
    else:
        with np.load(train_test_sample_list) as sample_ids:
            return sample_ids['train_ids'], sample_ids['test_ids']


def load_train_data(model_root_dir, total_samples_ls_fn, win_size,
                    min_r, min_f_deldup, min_f_neu, is_norm, test_ratio=0.3, min_size=100000):
    """

    :param model_root_dir:
    :param total_samples_ls_fn:
    :param win_size:
    :param min_r:
    :param min_f_deldup:
    :param min_f_neu:
    :param is_norm:
    :param min_sze:
    :return:
    """

    train_set_fn = os.path.join(model_root_dir,
                                'train{:.2f}_win{}_minsize{}_dataset.npz'.format(1-test_ratio, win_size, min_size))
    if os.path.exists(train_set_fn):
        with np.load(train_set_fn) as train_set:
            return train_set['x_train'], train_set['y_train']

    train_ids, _ = get_train_test_ids(model_root_dir, total_samples_ls_fn, test_ratio)
    x_train = []
    y_train_ = []
    for ix, train_id in enumerate(train_ids):
        x, y = get_data(train_id, model_root_dir, win_size, min_r, min_f_deldup, min_f_neu, is_norm)

        if (not (x is None)) and (not (y is None)):

            # deal with data balance
            del_idx = np.where(y == 'DEL')[0]
            dup_idx = np.where(y == 'DUP')[0]
            neu_idx = np.where(y == 'NEU')[0]

            len_del_idx = len(del_idx)
            len_dup_idx = len(dup_idx)
            len_neu_idx = len(neu_idx)

            if len_del_idx == 0 or len_dup_idx == 0 or len_neu_idx == 0:
                logger.warning('del len: {}, dup len: {}, neu len: {}'.format(len_del_idx, len_dup_idx, len_neu_idx))
                continue

            logger.info('del len: {}, dup len: {}, neu len: {}'.format(len_del_idx, len_dup_idx, len_neu_idx))
            min_idx_len = np.min([len(del_idx), len(dup_idx), len(neu_idx)])

            f_del_idx = np.random.choice(del_idx, min_idx_len, replace=False)
            f_dup_idx = np.random.choice(dup_idx, min_idx_len, replace=False)
            f_neu_idx = np.random.choice(neu_idx, min_idx_len, replace=False)

            f_idx = np.concatenate((f_del_idx, f_dup_idx, f_neu_idx))
            y_ = y[f_idx]
            x_ = x[f_idx, :, :]

            if len(x_train) == 0:
                x_train = x_
                y_train_ = y_
            else:
                x_train = np.concatenate((x_train, x_), axis=0)
                y_train_ = np.concatenate((y_train_, y_))

            if len(y_train_) >= 3*min_size:
                break
    y = [1 if x == 'DEL' else 2 if x == 'DUP' else 0 for x in y_train_]
    y_train = to_categorical(y)
    del y_train_
    gc.collect()
    logger.info('x train: {}, y train: {}'.format(x_train.shape, y_train.shape))
    np.savez_compressed(train_set_fn, x_train=x_train, y_train=y_train)
    return x_train, y_train