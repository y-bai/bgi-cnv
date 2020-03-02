#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: prediction_merge.py
    Description:
    
Created by Yong Bai on 2019/11/19 4:33 PM.
"""
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import argparse
import logging
from collections import Counter
import ruptures as rpt

from cnv_utils.general_utils import find_seg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


cnv_df = None
win_size = None
step_size =None
min_seg_len = None


def mp_init(l):
    global lock
    lock = l


def multi_run_wrapper(single_args):
    return get_prob_map(*single_args)


def get_prob_map(row_start_ind, row_end_ind):  # moving average
    # logger.info('row_start_ind: row_end_ind = {}:{}'.format(row_start_ind, row_end_ind))
    lock.acquire()

    seg_df = cnv_df.loc[row_start_ind: row_end_ind - 1].copy()

    prior_p = [0.7, 0.15, 0.15]
    seg_df['p_neu'] = seg_df['p_neu'] * prior_p[0]
    seg_df['p_del'] = seg_df['p_del'] * prior_p[1]
    seg_df['p_dup'] = seg_df['p_dup'] * prior_p[2]
    seg_df['t_sum'] = seg_df.loc[:, ['p_neu', 'p_del', 'p_dup']].sum(axis=1)
    seg_df['p_neu'] = seg_df['p_neu'] / seg_df['t_sum']
    seg_df['p_del'] = seg_df['p_del'] / seg_df['t_sum']
    seg_df['p_dup'] = seg_df['p_dup'] / seg_df['t_sum']
    seg_df.drop(columns=['t_sum'], inplace=True)

    re_start = []
    re_end = []
    re_merg_p_neu = []
    re_merg_p_del = []
    re_merg_p_dup = []
    re_merg_pre_l = []

    if row_end_ind - row_start_ind < min_seg_len:

        pos_s = seg_df.loc[row_start_ind, 'seg_s']
        re_start.append(pos_s)
        pos_e = seg_df.loc[row_end_ind-1, 'seg_s'] + step_size
        re_end.append(pos_e)
        merg_p_neu = seg_df.loc[row_start_ind:row_end_ind - 1, 'p_neu'].mean()
        re_merg_p_neu.append(merg_p_neu)
        merg_p_del = seg_df.loc[row_start_ind:row_end_ind - 1, 'p_del'].mean()
        re_merg_p_del.append(merg_p_del)
        merg_p_dup = seg_df.loc[row_start_ind:row_end_ind - 1, 'p_dup'].mean()
        re_merg_p_dup.append(merg_p_dup)
        # merg_p_lbl = np.argmax([merg_p_neu, merg_p_del, merg_p_dup])
        lbls_cnt = Counter(seg_df.loc[row_start_ind:row_end_ind - 1, 'pred_l'].values)
        merg_p_lbl = np.argmax([lbls_cnt.get(0, 0), lbls_cnt.get(1, 0), lbls_cnt.get(2, 0)])
        re_merg_pre_l.append(merg_p_lbl)

        lock.release()
        return re_start, re_end, re_merg_p_neu, re_merg_p_del, re_merg_p_dup, re_merg_pre_l
        # return None

    # seg_df['p_neu_win_avg'] = seg_df['p_neu'].rolling(int(win_size/step_size),
    #                                                   min_periods=0,
    #                                                   axis=0).mean()
    # seg_df['p_del_win_avg'] = seg_df['p_del'].rolling(int(win_size/step_size),
    #                                                   min_periods=0,
    #                                                   axis=0).mean()
    # seg_df['p_dup_win_avg'] = seg_df['p_dup'].rolling(int(win_size/step_size),
    #                                                   min_periods=0,
    #                                                   axis=0).mean()
    # prob_arr = seg_df[['p_neu_win_avg', 'p_del_win_avg', 'p_dup_win_avg']].values
    # prob_arr = seg_df[['p_neu', 'p_del', 'p_dup']].values
    prob_arr = seg_df[['p_del', 'p_dup']].values
    # prob_arr = seg_df['pred_l'].values
    # prob_arr = prob_arr/prob_arr.std(axis=0)

    # my_bkps = rpt.Binseg(model='l2', min_size=min_seg_len, jump=5).fit_predict(prob_arr, pen=10)
    # my_bkps = rpt.Binseg(model='l2', min_size=min_seg_len, jump=10).fit_predict(
    #     prob_arr, pen=5*np.log(prob_arr.shape[0]))
    my_bkps = rpt.Pelt(model='rbf', min_size=min_seg_len, jump=20).fit_predict(
        prob_arr, pen=10 * np.log(prob_arr.shape[0]))
    cp_bkps = row_start_ind + np.array([0] + my_bkps)

    for start_i, end_i in pairwise(cp_bkps):
        pos_s = seg_df.loc[start_i, 'seg_s']
        re_start.append(pos_s)
        pos_e = seg_df.loc[end_i-1, 'seg_s'] + step_size
        re_end.append(pos_e)

        # merg_p_neu = seg_df.loc[start_i:end_i - 1, 'p_neu_win_avg'].mean()
        # re_merg_p_neu.append(merg_p_neu)
        # merg_p_del = seg_df.loc[start_i:end_i - 1, 'p_del_win_avg'].mean()
        # re_merg_p_del.append(merg_p_del)
        # merg_p_dup = seg_df.loc[start_i:end_i - 1, 'p_dup_win_avg'].mean()
        # re_merg_p_dup.append(merg_p_dup)

        merg_p_neu = seg_df.loc[start_i:end_i-1, 'p_neu'].mean()
        re_merg_p_neu.append(merg_p_neu)
        merg_p_del = seg_df.loc[start_i:end_i-1, 'p_del'].mean()
        re_merg_p_del.append(merg_p_del)
        merg_p_dup = seg_df.loc[start_i:end_i-1, 'p_dup'].mean()
        re_merg_p_dup.append(merg_p_dup)

        # merg_p_lbl = np.argmax([merg_p_neu, merg_p_del, merg_p_dup])
        lbls_cnt = Counter(seg_df.loc[start_i:end_i - 1, 'pred_l'].values)
        merg_p_lbl = np.argmax([lbls_cnt.get(0, 0), lbls_cnt.get(1, 0), lbls_cnt.get(2, 0)])
        re_merg_pre_l.append(merg_p_lbl)

    lock.release()
    return re_start, re_end, re_merg_p_neu, re_merg_p_del, re_merg_p_dup, re_merg_pre_l


def pairwise(iterable):
    from itertools import tee
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def main(args):

    # input cnv call result
    global win_size
    win_size = args.win_size
    global step_size
    step_size = args.step_size

    in_fname = args.fname
    in_dir = args.i_root_dir
    out_dir = args.out_root_dir

    sample_id = args.sample_id
    chr_id = args.chr_id

    n_cpus = args.cpus

    out_type = args.out_type

    global min_seg_len
    min_seg_len = 5

    in_full_name = os.path.join(in_dir, in_fname)
    if not os.path.exists(in_full_name):
        raise FileNotFoundError('file not found {}'.format(in_full_name))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    global cnv_df
    cnv_df = pd.read_csv(in_full_name, sep='\t')
    cnv_df.loc[cnv_df['p_neu'].values.astype(int) == -1, ['p_neu', 'p_del', 'p_dup']] = np.nan

    # find predictive region

    cnv_df['pred_ind'] = np.where(cnv_df['indicator'] == 3, 1, 0)
    seg_values, seg_starts, seg_lengths = find_seg(cnv_df['pred_ind'].values)
    predictive_indices = np.where(seg_values == 1)[0]
    seg_start_pd_row_indices = seg_starts[predictive_indices]
    # index start at 0, end point should not include
    seg_end_pd_row_indices = seg_start_pd_row_indices + seg_lengths[predictive_indices]

    logger.info('segmenting and merging...')
    pred_seg_ind_lst = list(zip(seg_start_pd_row_indices, seg_end_pd_row_indices))
    len_segs = len(pred_seg_ind_lst)

    out_res = dict()
    start_pos = []
    end_pos = []
    merg_p_neu = []
    merg_p_del = []
    merg_p_dup = []
    merg_pre_l = []

    locker = mp.Lock()
    # with ThreadPool(n_proc) as p, h5py.File(online_out_sample_data_fn, 'w') as h5_out:
    with mp.Pool(n_cpus, initializer=mp_init, initargs=(locker,)) as p:

        results = p.imap(multi_run_wrapper, pred_seg_ind_lst)

        for i, res in enumerate(results):
            logger.info('finished at {}/{}'.format(i+1, len_segs))
            if res is None:
                logger.info('{}:{} cannot merge'.format(pred_seg_ind_lst[i][0], pred_seg_ind_lst[i][1]))
                continue
            re_start_i, re_end_i, re_merg_p_neu_i, re_merg_p_del_i, re_merg_p_dup_i, re_merg_pre_l_i = res
            # logger.info(re_start_i)

            start_pos.extend(re_start_i)
            end_pos.extend(re_end_i)
            merg_p_neu.extend(re_merg_p_neu_i)
            merg_p_del.extend(re_merg_p_del_i)
            merg_p_dup.extend(re_merg_p_dup_i)
            merg_pre_l.extend(re_merg_pre_l_i)

    out_res['POS_S'] = start_pos
    out_res['POS_E'] = end_pos
    out_res['LEN'] = np.array(end_pos) - np.array(start_pos)
    out_res['P_NEU'] = merg_p_neu
    out_res['P_DEL'] = merg_p_del
    out_res['P_DUP'] = merg_p_dup
    out_res['PRED_L'] = merg_pre_l

    out_cnv_df = pd.DataFrame(data=out_res)
    un_pred_df = cnv_df.loc[cnv_df['indicator'] != 3,
                            ['seg_s', 'seg_e', 'p_neu', 'p_del', 'p_dup', 'pred_l', 'indicator']]
    un_pred_df.loc[(cnv_df['indicator'] == 1) | (cnv_df['indicator'] == 2), 'seg_e'] =\
        un_pred_df.loc[(cnv_df['indicator'] == 1) | (cnv_df['indicator'] == 2), 'seg_s'] + 200
    un_pred_df['seg_len'] = un_pred_df['seg_e'] - un_pred_df['seg_s']
    un_pred_df = un_pred_df[['seg_s', 'seg_e', 'seg_len', 'p_neu', 'p_del', 'p_dup', 'pred_l']]

    whl_cnv_re = np.concatenate((out_cnv_df.values, un_pred_df.values), axis=0)
    ind = np.argsort(whl_cnv_re[:, 0])
    whl_cnv_re = whl_cnv_re[ind]
    f_out_df = pd.DataFrame(data=whl_cnv_re,
                          columns=['POS_S', 'POS_E', 'LEN', 'P_NEU', 'P_DEL', 'P_DUP', 'PRED_L'])
    f_out_df['POS_S'] = f_out_df['POS_S'].astype(int)
    f_out_df['POS_E'] = f_out_df['POS_E'].astype(int)
    f_out_df['LEN'] = f_out_df['LEN'].astype(int)
    f_out_df['PRED_L'] = f_out_df['PRED_L'].astype(int)

    # f_out_df = f_out_df[(f_out_df['PRED_L'] == 1) | (f_out_df['PRED_L'] == 2)]
    out_cnv_fn = os.path.join(out_dir, 'M{}_{}_{}_{}_out_cnv_{}-rbf_min5.csv'.format(
        sample_id, chr_id, win_size, step_size, out_type))
    if os.path.exists(out_cnv_fn):
        os.remove(out_cnv_fn)

    f_out_df.to_csv(out_cnv_fn, index=False, sep='\t')

    logger.info('Done, the results saved at {}'.format(out_cnv_fn))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cnv prediction merge ')

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
        "-p",
        "--out_type",
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