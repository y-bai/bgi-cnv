#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: mod_h5_chunk_size.py
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
    in_h5_fn = args.fname
    root_dir = args.root_dir

    i_h5_full = os.path.join(root_dir, in_h5_fn)
    o_h5_full = os.path.join(root_dir, 't_' + in_h5_fn)
    if os.path.exists(o_h5_full):
        os.remove(o_h5_full)

    logger.info('input file: {}'.format(i_h5_full))

    chk_size = 4  # 64*1000*13
    h5_str_dt = h5py.special_dtype(vlen=str)

    in_h5 = h5py.File(i_h5_full, 'r')
    x_dset = in_h5['x']
    y_dset = in_h5['str_y']

    n_smaples, win_size, n_features = x_dset.shape

    out_h5 = h5py.File(o_h5_full, 'w')
    out_xdset = out_h5.create_dataset('x',
                                      (n_smaples, win_size, n_features),
                                      maxshape=(None, win_size, n_features),
                                      dtype=np.float32,
                                      chunks=(chk_size, win_size, n_features),  # 512 * 8->256 for 100k win
                                      compression="gzip", compression_opts=4)

    out_xdset.attrs['n_sample'] = n_smaples
    out_ydset = out_h5.create_dataset('str_y',
                                      (n_smaples,),
                                      maxshape=(None,),
                                      dtype=h5_str_dt,
                                      chunks=(chk_size,),
                                      compression="gzip", compression_opts=4)

    logger.info('writing dataset...')

    slice_size = 256 * 16 * 4
    n_iter = int(n_smaples//slice_size)+1
    for i in range(n_iter):
        logger.info('finished at {}/{}'.format(i+1, n_iter+1))
        out_xdset[i*slice_size:(i+1)*slice_size] = x_dset[i*slice_size:(i+1)*slice_size]
        out_ydset[i*slice_size:(i+1)*slice_size] = y_dset[i*slice_size:(i+1)*slice_size]

    out_h5.close()
    in_h5.close()
    logger.info('Done, file saved at: {}'.format(o_h5_full))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create data sample list ')

    parser.add_argument(
        "-f",
        "--fname",
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
