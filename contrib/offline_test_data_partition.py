#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: offline_test_data_partition.py
    Description: since offline test dataset is huge,
    we first partition the whole dataset into smaller ones
    before run offline test.
    
Created by Yong Bai on 2019/9/30 3:44 PM.
"""
import os
import numpy as np
import pandas as pd
import h5py
import multiprocessing as mp
from multiprocessing import Manager
import ctypes

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mp_inti(l, test_fn_mp_list, test_lab_mp_list, part_size,
            out_dir, out_root_fn, win_size, n_feat):
    global lock
    lock = l

    global test_x_fn
    test_x_fn = test_fn_mp_list

    global test_y_lab
    test_y_lab = test_lab_mp_list

    global part_len
    part_len = part_size

    global out_dir_mp_g
    out_dir_mp_g = out_dir

    global out_root_fn_mp_g
    out_root_fn_mp_g = out_root_fn

    global win_len_g
    win_len_g = win_size

    global n_feat_g
    n_feat_g = n_feat


def load_feat(part_idx):
    # lock.acquire()
    i_test_fns = test_x_fn[part_idx * part_len.value:(part_idx + 1) * part_len.value]
    i_y_lab = test_y_lab[part_idx * part_len.value:(part_idx + 1) * part_len.value]

    len_out = len(i_test_fns)

    x_test = np.empty((len_out, win_len_g.value, n_feat_g.value), dtype=np.float32)
    for i, m_fn in enumerate(i_test_fns):
        with np.load(m_fn) as i_feat_map:
            x_test[i] = i_feat_map['x']
    out_fn = os.path.join(str(out_dir_mp_g.value.decode('utf-8')), str(out_root_fn_mp_g.value.decode('utf-8')))
    out_fn = out_fn + '-' + str(part_idx) + '-' + str(len_out) + '.h5'
    if os.path.exists(out_fn):
        os.remove(out_fn)
    with h5py.File(out_fn, 'w') as out_h5:
        out_h5.create_dataset('x_test', data=x_test, compression="gzip", compression_opts=4)
        out_h5.create_dataset('y_test', data=i_y_lab, compression="gzip", compression_opts=4)
    # lock.release()
    return 'finished at {}, saved into {}'.format(part_idx, out_fn)


def main(in_test_data_fn, out_dir, win_size=1000, n_feat=13, min_r=0.1, min_f=0.01, n_proc=16):

    logger.info('loading test data set....')
    test_sample_df = pd.read_csv(in_test_data_fn, sep='\t')
    test_samples_fn_arr = test_sample_df['f_name'].values
    test_samples_true_arr = test_sample_df['cnv_type_encode'].values

    out_root_fn = 'w{0}_r{1:.2f}_f{2:.2f}'.format(win_size, min_r, min_f)
    test_sample_size = len(test_samples_true_arr)
    logger.info('the total number of test instance {}'.format(test_sample_size))
    part_size = 1024 * 50
    n_parts = int(np.floor(test_sample_size / float(part_size)))

    logger.info('the total number of test instance {}, parts {}'.format(test_sample_size, n_parts))

    mp_manager = Manager()
    test_fn_mp_list = mp_manager.list(test_samples_fn_arr)
    test_lab_mp_list = mp_manager.list(test_samples_true_arr)

    part_mp = mp.Value('i', part_size)
    out_dir_mp = mp.Value(ctypes.c_char_p, bytes(out_dir.encode('utf-8')))
    out_root_fn_mp = mp.Value(ctypes.c_char_p, bytes(out_root_fn.encode('utf-8')))

    win_size_mp = mp.Value('i', win_size)
    n_feat_mp = mp.Value('i', n_feat)

    locker = mp.Lock()
    logger.info('the number of cpu {}'.format(n_proc))
    with mp.Pool(n_proc, initializer=mp_inti,
                 initargs=(locker, test_fn_mp_list, test_lab_mp_list, part_mp,
                           out_dir_mp, out_root_fn_mp, win_size_mp, n_feat_mp)) as p:
        # if computation intensive, then it should not use chunksize otherwise the process will be blocked by os.
        results = p.imap(load_feat, list(range(n_parts)))

        for res in results:
            logger.info(res)

    logger.info('partion Done, saved in dir {}'.format(out_dir))


if __name__ == '__main__':

    in_test_data_fn = '/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data/w1000_r0.10_f0.01_test.csv'
    if not os.path.exists(in_test_data_fn):
        raise FileNotFoundError('test file not found {}'.format(in_test_data_fn))

    out_root_dir = '/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data'
    out_sub_dir = os.path.join(out_root_dir, 'partition_test_features')
    if not os.path.isdir(out_sub_dir):
        os.mkdir(out_sub_dir)

    win_size = 1000
    n_feat = 13
    min_r = 0.1
    min_f = 0.01
    n_proc = mp.cpu_count()

    main(in_test_data_fn, out_sub_dir,
         win_size=win_size, n_feat=n_feat, min_r=min_r, min_f=min_f, n_proc=n_proc)
