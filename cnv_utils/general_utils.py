#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: general_utils.py
    Description:
    
Created by Yong Bai on 2019/8/13 10:45 AM.
"""
import numpy as np


def str2bool(v):
    """
    convert str to bool, which is used in args
    :param v: input str to be convert to bool
    :return:
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')


def find_seg(x):
    """
    ref: https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    :param x:
    :return:
    """

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find seg starts
        loc_seg_start = np.empty(n, dtype=bool)
        loc_seg_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_seg_start[1:])
        seg_starts = np.nonzero(loc_seg_start)[0]

        # find seg values
        #
        seg_values = x[loc_seg_start]

        # find seg lengths
        seg_lengths = np.diff(np.append(seg_starts, n))

        return seg_values, seg_starts, seg_lengths


def seq_slide(seq_len, win_size, stride_size):
    """

    :param seq_len:
    :param win_size:
    :param stride_size:
    :return:
    """
    if seq_len < win_size:
        raise Exception("length of sequence less than win size when slide window.")
    n_win = int((seq_len - win_size) // stride_size) + 1

    seq_start_indices = range(0, n_win * stride_size, stride_size)
    end_start = np.max(seq_start_indices) + win_size
    remain_len = seq_len - end_start
    return np.array(seq_start_indices, dtype=int), end_start, remain_len
