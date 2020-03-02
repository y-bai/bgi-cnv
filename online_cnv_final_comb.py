#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: online_cnv_final_comb.py
    Description:
    
Created by Yong Bai on 2019/12/16 9:38 PM.
"""
import os
import numpy as np
import pandas as pd
import logging
import h5py
import argparse


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def rm_duprows(df):
    pred_lbl = df['PRED_P_L'].value_counts().keys().tolist()
    if len(pred_lbl) > 1 and 0 in pred_lbl:
        return df[df['PRED_P_L'] != 0]
    elif len(pred_lbl) == 1 and pred_lbl[0] == 0:
        return df[0:1]
    else:
        return df


def cal_fp_tp_fn(sample_analysis_fn, reg_size):

    in_out_pred = pd.read_csv(sample_analysis_fn, sep='\t')
    tmp_df = in_out_pred[['P_NEU', 'P_DEL', 'P_DUP']]
    tmp_df.columns = ['0', '1', '2']
    in_out_pred['PRED_P_L'] = tmp_df.idxmax(axis=1, skipna=True)
    in_out_pred['PRED_P_L'] = in_out_pred['PRED_P_L'].astype(int)

    # Index(['POS_S', 'POS_E', 'LEN', 'P_NEU', 'P_DEL', 'P_DUP', 'PRED_L', 'ID',
    #        'POS', 'END', 'SVLEN', 'ALT', 'SMP_GT', 'ADJTYPE', 'ADJAF',
    #        'INTERVAL_LOWER', 'INTERVAL_UPPER', 'INTERVAL_LEN', 'UNION_LEN',
    #        'CHROM', 'TRUE_L', 'CENTOR_INTER_LEN', 'SUPDUP_INTER_LEN',
    #        'CENT_CALLLEN_RATIO', 'SUPDUP_CALLLEN_RATIO', 'PRED_P_L'],
    #       dtype='object')

    in_pred_analysis_df = in_out_pred[['POS_S', 'POS_E', 'LEN', 'P_NEU', 'P_DEL', 'P_DUP', 'ID',
                                       'POS', 'END', 'SVLEN', 'ALT', 'SMP_GT',
                                       'CHROM', 'INTERVAL_LEN', 'UNION_LEN', 'TRUE_L', 'PRED_P_L']].copy()

    in_pred_analysis_df.drop(in_pred_analysis_df[(in_pred_analysis_df['TRUE_L'] == 0) &
                                                 (in_pred_analysis_df['PRED_P_L'] == 0)].index, inplace=True)

    f_fps_df = in_pred_analysis_df[(in_pred_analysis_df['TRUE_L'] == 0) &
                                   (in_pred_analysis_df['PRED_P_L'] != 0)]

    # tp + fn = all positive
    positive_res_df = in_pred_analysis_df[(in_pred_analysis_df['TRUE_L'] != 0)]
    gp_pos_df = positive_res_df.groupby('ID')
    filtered_gp_pos_df = gp_pos_df.apply(rm_duprows).reset_index(drop=True)

    all_tp = []
    all_fp = []
    all_real_pos = []

    del_tp = []
    del_fp = []
    del_real_pos = []

    dup_tp = []
    dup_fp = []
    dup_real_pos = []

    for i_reg in reg_size:
        i_all_fp_df = f_fps_df[f_fps_df['LEN'] >= i_reg]
        i_del_fp_df = f_fps_df[(f_fps_df['LEN'] >= i_reg) & (f_fps_df['PRED_P_L'] == 1)]
        i_dup_fp_df = f_fps_df[(f_fps_df['LEN'] >= i_reg) & (f_fps_df['PRED_P_L'] == 2)]
        all_fp.append(len(i_all_fp_df))
        del_fp.append(len(i_del_fp_df))
        dup_fp.append(len(i_dup_fp_df))

        all_pos_pf_df = filtered_gp_pos_df[(filtered_gp_pos_df['LEN'] >= i_reg)]
        del_pos_pf_df = filtered_gp_pos_df[(filtered_gp_pos_df['LEN'] >= i_reg) &
                                       (filtered_gp_pos_df['TRUE_L'] == 1)]
        dup_pos_pf_df = filtered_gp_pos_df[(filtered_gp_pos_df['LEN'] >= i_reg) &
                                       (filtered_gp_pos_df['TRUE_L'] == 2)]
        all_real_pos.append(len(all_pos_pf_df))
        del_real_pos.append(len(del_pos_pf_df))
        dup_real_pos.append(len(dup_pos_pf_df))

        all_tp_df = all_pos_pf_df[all_pos_pf_df['PRED_P_L'] == all_pos_pf_df['TRUE_L']]
        del_tp_df = del_pos_pf_df[del_pos_pf_df['PRED_P_L'] == del_pos_pf_df['TRUE_L']]
        dup_tp_df = dup_pos_pf_df[dup_pos_pf_df['PRED_P_L'] == dup_pos_pf_df['TRUE_L']]
        all_tp.append(len(all_tp_df))
        del_tp.append(len(del_tp_df))
        dup_tp.append(len(dup_tp_df))

    all_re_df = pd.DataFrame(data={'tp': all_tp, 'fp': all_fp, 'tp_fn': all_real_pos, 'range_low': reg_size})
    del_re_df = pd.DataFrame(data={'tp': del_tp, 'fp': del_fp, 'tp_fn': del_real_pos, 'range_low': reg_size})
    dup_re_df = pd.DataFrame(data={'tp': dup_tp, 'fp': dup_fp, 'tp_fn': dup_real_pos, 'range_low': reg_size})

    return all_re_df[['range_low', 'tp', 'fp', 'tp_fn']].values, \
           del_re_df[['range_low', 'tp', 'fp', 'tp_fn']].values, \
           dup_re_df[['range_low', 'tp', 'fp', 'tp_fn']].values


def main(args):

    win_size = args.win_size
    step_size = args.step_size

    sample_id_str = args.sample_ids_lst
    sample_ids = sample_id_str.split(' ')
    n_samples = len(sample_ids)
    reg_size_lst = args.reg_size_lst
    reg_size = [int(x) for x in reg_size_lst.split(' ')]
    n_reg = len(reg_size)

    all_tp_fp_fn_lst = []
    del_tp_fp_fn_lst = []
    dup_tp_fp_fn_lst = []
    for sample_id in sample_ids:

        logger.info('processing {}'.format(sample_id))
        # i_in_dir = '/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/1KGP_bak/'+sample_id+'/cnv_call'
        i_in_dir = '/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/1KGP/win_' + str(win_size) + '/' + sample_id + '/cnv_call'
        i_ana_fn = os.path.join(i_in_dir, 'Analysis{}_{}_{}_out_cnv.csv'.format(sample_id, win_size, step_size))

        if not os.path.exists(i_ana_fn):
            raise FileNotFoundError('file does not exist. {}'.format(i_ana_fn))

        all_re, del_re, dup_re = cal_fp_tp_fn(i_ana_fn, reg_size)
        all_tp_fp_fn_lst.append(all_re)
        del_tp_fp_fn_lst.append(del_re)
        dup_tp_fp_fn_lst.append(dup_re)

    all_tp_fp_fn_arr = np.array(all_tp_fp_fn_lst)  # list of 2d array to 3d array
    del_tp_fp_fn_arr = np.array(del_tp_fp_fn_lst)
    dup_tp_fp_fn_arr = np.array(dup_tp_fp_fn_lst)

    logger.info('tp_fp_fn array shape: {}'.format(all_tp_fp_fn_arr.shape))

    out_fn = '/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/1KGP/win_{}/ns{}_nr{}.h5'.format(
        win_size, n_samples, n_reg)
    if os.path.exists(out_fn):
        os.remove(out_fn)

    with h5py.File(out_fn, 'w') as h5_out:
        h5_out.create_dataset('all_tp_fp_fn', data=all_tp_fp_fn_arr)
        h5_out.create_dataset('del_tp_fp_fn', data=del_tp_fp_fn_arr)
        h5_out.create_dataset('dup_tp_fp_fn', data=dup_tp_fp_fn_arr)
    logger.info('Done, result saving at {}'.format(out_fn))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cnv combination analysis ')

    parser.add_argument(
        "-s",
        "--sample_ids_lst",
        type=str)

    parser.add_argument(
        "-r",
        "--reg_size_lst",
        type=str)

    parser.add_argument(
        "-w",
        "--win_size",
        type=int)

    parser.add_argument(
        "-p",
        "--step_size",
        type=int)

    args = parser.parse_args()
    logger.info('args: {}'.format(args))
    main(args)