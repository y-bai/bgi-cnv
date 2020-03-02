#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: online_cnv_call_eval_main.py
    Description:
    
Created by Yong Bai on 2019/11/19 4:33 PM.
"""
import os
import numpy as np
import multiprocessing as mp
import pandas as pd
import argparse
import logging
import intervals as I

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


true_cnv_df = None
n_true_cnv_df_cols = None


def mp_init(l):
    global lock
    lock = l


def cnv_pred_true_merge(rowiter):
    lock.acquire()
    idx, row = rowiter
    res = []
    for i_idx, i_row in true_cnv_df.iterrows():
        x = I.closed(row['POS_S'], row['POS_E']) & I.closed(i_row['POS'], i_row['END'])
        y = I.closed(row['POS_S'], row['POS_E']) | I.closed(i_row['POS'], i_row['END'])
        if x.is_empty():
            continue
        inter_len = x.upper - x.lower
        union_len = y.upper - y.lower
        if inter_len >= 0:
            res.append(np.concatenate((row.values, i_row.values, [x.lower, x.upper, inter_len, union_len]), axis=None))
    if len(res) == 0:
        res.append(np.concatenate((row.values, [np.nan]*n_true_cnv_df_cols, [-1]*4), axis=None))
    lock.release()
    return res


def main(args):

    # input cnv call result
    win_size = args.win_size
    step_size = args.step_size

    in_fname = args.fname
    in_dir = args.i_root_dir
    out_dir = args.out_root_dir

    sample_id = args.sample_id
    chr_id = args.chr_id
    n_cpus = args.cpus

    truth_cnv_root_dir = args.truth_cnv_root_dir
    truth_cnv_label_fn = os.path.join(truth_cnv_root_dir,
                                      'ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.{}.cnvs.labels'.format(sample_id))

    in_full_name = os.path.join(in_dir, in_fname)
    if not os.path.exists(in_full_name):
        raise FileNotFoundError('file not found {}'.format(in_full_name))

    if not os.path.exists(truth_cnv_label_fn):
        raise FileNotFoundError('file not found {}'.format(truth_cnv_label_fn))

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    pred_cnv_df = pd.read_csv(in_full_name, sep='\t')
    pred_cnv_df['POS_S'] = pred_cnv_df['POS_S'].astype(int)
    pred_cnv_df['POS_E'] = pred_cnv_df['POS_E'].astype(int)
    pred_cnv_df['LEN'] = pred_cnv_df['LEN'].astype(int)
    pred_cnv_df['PRED_L'] = pred_cnv_df['PRED_L'].astype(int)

    # remove undetectable region
    pred_cnv_df = pred_cnv_df[pred_cnv_df['PRED_L'] != -1]

    n_pred_cnv = pred_cnv_df.shape[0]
    t_true_cnv_df = pd.read_csv(truth_cnv_label_fn, sep='\t')
    global true_cnv_df
    true_cnv_df = t_true_cnv_df[t_true_cnv_df['CHROM'] == chr_id]
    true_cnv_df_cols = list(true_cnv_df.columns)
    global n_true_cnv_df_cols
    n_true_cnv_df_cols = len(true_cnv_df_cols)

    f_cols = list(pred_cnv_df.columns) + true_cnv_df_cols + ['INTERVAL_LOWER', 'INTERVAL_UPPER', 'INTERVAL_LEN', 'UNION_LEN']

    f_out_lst = []
    locker = mp.Lock()
    # with ThreadPool(n_proc) as p, h5py.File(online_out_sample_data_fn, 'w') as h5_out:
    with mp.Pool(n_cpus, initializer=mp_init, initargs=(locker,)) as p:
        results = p.imap(cnv_pred_true_merge, pred_cnv_df.iterrows())
        for i, i_res in enumerate(results):
            logger.info('finished at {}/{}'.format(i+1, n_pred_cnv))
            f_out_lst.append(np.vstack(i_res))

    f_out_arr = np.vstack(f_out_lst)
    f_out_df = pd.DataFrame(data=f_out_arr, columns=f_cols)
    f_out_df['POS_S'] = f_out_df['POS_S'].astype(int)
    f_out_df['POS_E'] = f_out_df['POS_E'].astype(int)
    f_out_df['LEN'] = f_out_df['LEN'].astype(int)
    f_out_df['PRED_L'] = f_out_df['PRED_L'].astype(int)

    f_out_df['INTERVAL_LOWER'] = f_out_df['INTERVAL_LOWER'].astype(int)
    f_out_df['INTERVAL_UPPER'] = f_out_df['INTERVAL_UPPER'].astype(int)
    f_out_df['INTERVAL_LEN'] = f_out_df['INTERVAL_LEN'].astype(int)
    f_out_df['UNION_LEN'] = f_out_df['UNION_LEN'].astype(int)

    out_cnv_fn = os.path.join(out_dir, 'Eval{}_{}_{}_{}_out_cnv.csv'.format(sample_id, chr_id, win_size, step_size))
    if os.path.exists(out_cnv_fn):
        os.remove(out_cnv_fn)
    f_out_df.to_csv(out_cnv_fn, index=False, sep='\t')

    logger.info('Done, the results saved at {}'.format(out_cnv_fn))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cnv evaluation ')

    parser.add_argument(
        "-d",
        "--truth_cnv_root_dir",
        type=str)

    parser.add_argument(
        "-w",
        "--win_size",
        type=int)

    parser.add_argument(
        "-s",
        "--step_size",
        type=int)

    parser.add_argument(
        "-m",
        "--sample_id",
        type=str)

    parser.add_argument(
        "-c",
        "--chr_id",
        type=str)

    parser.add_argument(
        "-t",
        "--cpus",
        type=int)

    parser.add_argument(
        "-f",
        "--fname",
        type=str)

    parser.add_argument(
        "-i",
        "--i_root_dir",
        type=str,
        help="in directory")

    parser.add_argument(
        "-o",
        "--out_root_dir",
        type=str,
        help="output directory")

    args = parser.parse_args()
    logger.info('args: {}'.format(args))
    main(args)