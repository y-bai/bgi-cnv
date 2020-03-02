#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: online_cnv_eval_analysis.py
    Description:
    
Created by Yong Bai on 2019/11/28 9:13 AM.
"""

import os
import numpy as np
import multiprocessing as mp
import pandas as pd
import glob
import argparse
import logging
import intervals as I

from cnv_utils import load_centro, load_sup_dups

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

centros_regions_df = None
sup_dups_regions_df = None


def mp_init(l):
    global lock
    lock = l


def eval_cnv_wrapper(func_param):
    return eval_cnv(*func_param)


def eval_cnv(chr_id, chr_i_eval_fname, chr_centro_df, chr_supdups_df):
    lock.acquire()

    chr_i_eval_df = pd.read_csv(chr_i_eval_fname, sep='\t')
    cols = ['POS_S', 'POS_E', 'LEN', 'P_NEU', 'P_DEL', 'P_DUP', 'PRED_L',
            'ID', 'POS', 'END', 'SVLEN', 'ALT', 'SMP_GT', 'ADJTYPE',
            'ADJAF', 'INTERVAL_LOWER', 'INTERVAL_UPPER', 'INTERVAL_LEN',
            'UNION_LEN']

    chr_i_eval_df = chr_i_eval_df[cols]

    # find the positive(tp + fp) and false negative
    pred_cnv_df = chr_i_eval_df[((~chr_i_eval_df['ID'].isnull()) |
                             (chr_i_eval_df['PRED_L'] == 1) |
                             (chr_i_eval_df['PRED_L'] == 2)) &
                            (chr_i_eval_df['PRED_L'] != -1)].copy().reset_index(drop=True)

    del chr_i_eval_df
    pred_cnv_df['ADJAF'] = pred_cnv_df['ADJAF'].astype(float)

    pred_cnv_df['CHROM'] = 'chr' + str(chr_id)
    pred_cnv_df['TRUE_L'] = 0
    if pred_cnv_df['ADJTYPE'].isin(['DUP']).any():
        pred_cnv_df.loc[pred_cnv_df['ADJTYPE'] == 'DUP', 'TRUE_L'] = 2
    if pred_cnv_df['ADJTYPE'].isin(['DEL']).any():
        pred_cnv_df.loc[pred_cnv_df['ADJTYPE'] == 'DEL', 'TRUE_L'] = 1

    pred_cnv_df['CENTOR_INTER_LEN'] = 0
    pred_cnv_df['SUPDUP_INTER_LEN'] = 0
    pred_cnv_df['CENT_CALLLEN_RATIO'] = 0.0
    pred_cnv_df['SUPDUP_CALLLEN_RATIO'] = 0.0
    for r_idx, r_row in pred_cnv_df.iterrows():
        # find centors regions overlap
        for c_idx, c_row in chr_centro_df.iterrows():
            x_cent = I.closed(r_row['POS_S'], r_row['POS_E']) & I.closed(c_row['START'], c_row['END'])
            if x_cent.is_empty():
                r_row['CENTOR_INTER_LEN'] = 0
            else:
                r_row['CENTOR_INTER_LEN'] = x_cent.upper - x_cent.lower

        # find sup dups regions overlap
        for s_idx, s_row in chr_supdups_df.iterrows():
            x_supdup = I.closed(r_row['POS_S'], r_row['POS_E']) & I.closed(s_row['START'], s_row['END'])
            if x_supdup.is_empty():
                r_row['SUPDUP_INTER_LEN'] = 0
            else:

                r_row['SUPDUP_INTER_LEN'] = x_supdup.upper - x_supdup.lower

    pred_cnv_df['CENT_CALLLEN_RATIO'] = pred_cnv_df['CENTOR_INTER_LEN'] / pred_cnv_df['LEN']
    pred_cnv_df['SUPDUP_CALLLEN_RATIO'] = pred_cnv_df['SUPDUP_INTER_LEN'] / pred_cnv_df['LEN']


    # temp filters
    # pred_cnv_df.loc[(np.abs(pred_cnv_df['P_DUP'] - pred_cnv_df['P_NEU']) < 0.1) &
    #                 (pred_cnv_df['PRED_L'] == 2), 'PRED_L'] = 0
    # pred_cnv_df.loc[(np.abs(pred_cnv_df['P_DEL'] - pred_cnv_df['P_NEU']) < 0.1) &
    #                 (pred_cnv_df['PRED_L'] == 1), 'PRED_L'] = 0
    # # 2 reduce fp
    # tmp_df = pred_cnv_df[['P_NEU', 'P_DEL', 'P_DUP']]
    # tmp_df.columns = ['0', '1', '2']
    # pred_cnv_df['PRED_P_L'] = tmp_df.idxmax(axis=1, skipna=True)
    # pred_cnv_df['PRED_P_L'] = pred_cnv_df['PRED_P_L'].astype(int)
    # pred_cnv_df.loc[(pred_cnv_df['PRED_L'] != 0) & (pred_cnv_df['PRED_P_L'] == 0), 'PRED_L'] = 0

    lock.release()
    return pred_cnv_df


def main(args):

    # input cnv call result
    win_size = args.win_size
    step_size = args.step_size

    in_dir = args.i_root_dir
    out_dir = args.out_root_dir

    sample_id = args.sample_id
    n_cpus = args.cpus

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    centros_regions_df = load_centro(centro_fn='../hg38_config/hg38_centromeres.txt')
    sup_dups_regions_df = load_sup_dups(sup_dups_fn='../hg38_config/hg38_centromeres.txt')

    def getkey(full_filename):
        f_text_name = os.path.splitext(os.path.basename(full_filename))
        return int(f_text_name[0].split('_')[1])
    # get Eval files
    fnames = os.path.join(in_dir,
                          'Eval{0}_*_{1}_{2}_out_cnv.csv'.format(sample_id, win_size, step_size))
    eval_files = glob.glob(fnames)

    eval_files = sorted(eval_files, key=getkey)
    # pred_cnv_lens = [(1000, 10*1000), (10*1000, 50*1000),
    #                  (50*1000, 100*1000), (100*1000, 500*1000), (500*1000, np.inf)]
    eval_cnv_lst = []
    for i_fname in eval_files:
        f_t_name = os.path.splitext(os.path.basename(i_fname))
        chr_id = f_t_name[0].split('_')[1]
        chr_centro_df = centros_regions_df[centros_regions_df['CHROM'] == 'chr'+chr_id]
        chr_supdups_df = sup_dups_regions_df[sup_dups_regions_df['CHROM'] == 'chr'+chr_id]
        eval_cnv_lst.append((chr_id, i_fname, chr_centro_df, chr_supdups_df))

    logger.info('eval_cnv_lst: {}'.format(len(eval_cnv_lst)))
    out_cnv_df = None
    locker = mp.Lock()
    # with ThreadPool(n_proc) as p, h5py.File(online_out_sample_data_fn, 'w') as h5_out:
    with mp.Pool(n_cpus, initializer=mp_init, initargs=(locker,)) as p:
        results = p.imap(eval_cnv_wrapper, eval_cnv_lst)
        for k, i_res in enumerate(results):
            logger.info('finshed {}, shape:{}'.format(k, i_res.shape))
            if out_cnv_df is None:
                out_cnv_df = i_res
            else:
                out_cnv_df = pd.concat([out_cnv_df, i_res], ignore_index=True)

    out_cnv_fn = os.path.join(out_dir, 'Analysis{}_{}_{}_out_cnv.csv'.format(sample_id, win_size, step_size))
    if os.path.exists(out_cnv_fn):
        os.remove(out_cnv_fn)
    if out_cnv_df is not None:
        out_cnv_df.to_csv(out_cnv_fn, index=False, sep='\t')
        logger.info('Done, the results saved at {}'.format(out_cnv_fn))
    else:
        logger.info('The output dataframe is None, please check')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cnv evaluation ')

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
        "-t",
        "--cpus",
        type=int)

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