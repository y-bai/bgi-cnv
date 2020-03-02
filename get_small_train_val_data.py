#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: get_small_train_val_data.py
    Description:

Created by Yong Bai on 2019/11/3 10:51 AM.
"""
import os
import numpy as np
import pandas as pd


def _sampling_data(data_df, out_fn, frac=0.3):

    del_samples = data_df[data_df['cnv_type_encode'] == 1]
    dup_samples = data_df[data_df['cnv_type_encode'] == 2]
    neu_samples = data_df[data_df['cnv_type_encode'] == 0]

    n_sample = int(len(del_samples)*frac)

    # random sample the samples to make the balance data set
    sel_del_samples = del_samples.sample(n=n_sample, random_state=123).reset_index(drop=True)
    sel_dup_samples = dup_samples.sample(n=n_sample, random_state=123).reset_index(drop=True)
    sel_neu_samples = neu_samples.sample(n=n_sample, random_state=123).reset_index(drop=True)

    f_samples = pd.concat([sel_del_samples, sel_dup_samples, sel_neu_samples], ignore_index=True)
    # shuffle rows
    f_samples = f_samples.sample(frac=1).reset_index(drop=True)
    out_fn = out_fn + 'tmp_f{}'.format(frac)
    f_samples.to_csv(out_fn, sep='\t', index=False)


if __name__ == '__main__':
    # this is used to generate samll dataset to train model fast
    origin_data_dir = '/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data'
    origin_data_train_fn = os.path.join(origin_data_dir, 'w1000_r0.10_f0.01_train_train_111.csv')
    origin_data_val_fn = os.path.join(origin_data_dir, 'w1000_r0.10_f0.01_train_val_111.csv')

    origin_data_train_df = pd.read_csv(origin_data_train_fn, '\t')  # n_neu = n_del = n_dup = 1933713
    origin_data_val_df = pd.read_csv(origin_data_val_fn, '\t')  # n_neu = n_del = n_dup = 484033

    _sampling_data(origin_data_train_df, origin_data_train_fn)
    _sampling_data(origin_data_val_df, origin_data_val_fn)





