#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: merge_train_val_data_h5.py
    Description:
    
Created by Yong Bai on 2019/11/14 5:24 PM.
"""

import os
import numpy as np
import argparse
import logging
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(args):
    in_train_fn = args.train_fn
    in_val_fn = args.val_fn

    root_dir = args.root_dir

    in_train_h5_fn = os.path.join(root_dir, in_train_fn)
    in_val_h5_fn = os.path.join(root_dir, in_val_fn)

    o_h5_full = os.path.join(root_dir, 'whole_train_val.h5')

    if os.path.exists(o_h5_full):
        os.remove(o_h5_full)

    logger.info('input file: {}\n{}'.format(in_train_h5_fn, in_val_h5_fn))

    in_train_h5 = h5py.File(in_train_h5_fn, 'r')
    x_train_dset = in_train_h5['x']
    y_train_dset = in_train_h5['y']

    in_val_h5 = h5py.File(in_val_h5_fn, 'r')
    x_val_dset = in_val_h5['x']
    y_val_dset = in_val_h5['y']

    n_train_smaples, win_size_train, n_features_train = x_train_dset.shape
    n_val_smaples, win_size_val, n_features_val = x_val_dset.shape

    assert win_size_train == win_size_val
    assert n_features_train == n_features_val

    logger.info('train data shape: x={}, y={}'.format(x_train_dset.shape, y_train_dset.shape))
    logger.info('val data shape: x={}, y={}'.format(x_val_dset.shape, y_val_dset.shape))

    out_samples = n_train_smaples + n_val_smaples
    chk_size = 2  # 64*1000*13

    out_h5 = h5py.File(o_h5_full, 'w')
    out_xdset = out_h5.create_dataset('x',
                                      (out_samples, win_size_train, n_features_train),
                                      maxshape=(None, win_size_train, n_features_train),
                                      dtype=np.float32,
                                      chunks=(chk_size, win_size_train, n_features_train),  # 512 * 8->256 for 100k win
                                      compression="gzip", compression_opts=4)

    out_ydset = out_h5.create_dataset('y',
                                      (out_samples,),
                                      maxshape=(None,),
                                      dtype=np.int32,
                                      chunks=(chk_size,),
                                      compression="gzip", compression_opts=4)

    logger.info('writing dataset...')

    slice_size = 256 * 16 * 4
    n_iter_train = int(n_train_smaples//slice_size)+1
    for i in range(n_iter_train):
        logger.info('train data finished at {}/{}'.format(i+1, n_iter_train+1))
        out_xdset[i*slice_size:(i+1)*slice_size] = x_train_dset[i*slice_size:(i+1)*slice_size]
        out_ydset[i*slice_size:(i+1)*slice_size] = y_train_dset[i*slice_size:(i+1)*slice_size]

    logger.info('out with train shape: x={}, y={}'.format(out_xdset.shape, out_ydset.shape))

    n_iter_val = int(n_val_smaples // slice_size) + 1
    for i in range(n_iter_val):
        logger.info('val data finished at {}/{}'.format(i + 1, n_iter_val + 1))
        out_xdset[(n_train_smaples + i * slice_size): (n_train_smaples + (i + 1) * slice_size)] = \
            x_val_dset[i * slice_size:(i + 1) * slice_size]
        out_ydset[(n_train_smaples + i * slice_size): (n_train_smaples + (i + 1) * slice_size)] = \
            y_val_dset[i * slice_size:(i + 1) * slice_size]

    out_xdset.attrs['n_sample'] = n_train_smaples + n_val_smaples

    logger.info('out with train and val shape: x={}, y={}'.format(out_xdset.shape, out_ydset.shape))

    out_h5.close()
    in_val_h5.close()
    in_train_h5.close()

    logger.info('Done, file saved at: {}'.format(o_h5_full))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create data sample list ')

    parser.add_argument(
        "-t",
        "--train_fn",
        type=str)

    parser.add_argument(
        "-v",
        "--val_fn",
        type=str)

    parser.add_argument(
        "-d",
        "--root_dir",
        type=str,
        help="output directory",
        required=True)

    args = parser.parse_args()
    logger.info('args: {}'.format(args))
    main(args)
