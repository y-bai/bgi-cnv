#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: model_train_main.py
    Description:
    
Created by Yong Bai on 2019/8/20 3:13 PM.
"""

import os
from train_model import train

if __name__ == "__main__":
    # ref:
    # https://github.com/keras-team/keras/issues/10340
    # https://github.com/keras-team/keras/issues/11671
    # https://github.com/keras-team/keras/issues/10948
    # mp.set_start_method('spawn')
    # https://docs.python.org/3.4/library/multiprocessing.html?highlight=process#multiprocessing.set_start_method
    # mp.set_start_method('forkserver')  # thread-safe

    # path for save the model
    out_root_dir = "/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out"
    in_data_root_dir = '/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data'

    # in_data_root_dir = '/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/npz_feature_file/NIPT_like_1000genome_2504_win500_train_feature'

    win_size = 1000  # 1000->500
    min_r = 0.1
    min_f = 0.01
    nb_cls_prop = '1:1:1'
    # nb_cls_prop = '1:2:3'

    penalty_weights_index = 1  # this is used as identification of cost

    # Note: class_weight in fit_generator would course OOM
    # https://github.com/tensorflow/tensorflow/issues/31253
    # class_weights = None

    train_sample_fn = os.path.join(in_data_root_dir,
                                   'w{0}_r{1:.2f}_f{2:.2f}_train_train{3}.csv'.format(
                                    win_size, min_r, min_f, '_'+''.join(nb_cls_prop.split(':'))))
    val_sample_fn = os.path.join(in_data_root_dir,
                                 'w{0}_r{1:.2f}_f{2:.2f}_train_val{3}.csv'.format(
                                  win_size, min_r, min_f, '_' + ''.join(nb_cls_prop.split(':'))))

    out_model_root_dir = os.path.join(out_root_dir, 'model{}'.format(''.join(nb_cls_prop.split(':'))))

    if not os.path.exists(train_sample_fn):
        raise FileNotFoundError('train sample file not found: {}'.format(train_sample_fn))
    if not os.path.exists(val_sample_fn):
        raise FileNotFoundError('validation sample file not found: {}'.format(val_sample_fn))
    if not os.path.isdir(out_model_root_dir):
        os.mkdir(out_model_root_dir)

    # batch_size = 512 16-2 see the difference on class_weight
    # model_params = {'n_win_size': win_size,
    #                 'n_feat': 13,
    #                 'n_class': 3,
    #                 'epochs': 50,
    #                 'batch': 512,
    #                 'learn_rate': 0.001,
    #                 'drop': 0.1,
    #                 'fc_size': 32,
    #                 'blocks': '4_4_1',
    #                 'n_gpu': 4,
    #                 'n_cpu': 20,   #
    #                 'class_weights': class_weights}
    # batch_size = 1024 15-1 old parameters
    # batch 512
    model_params = {'n_win_size': win_size,
                    'n_feat': 13,
                    'n_class': 3,
                    'epochs': 64,
                    'batch': 512,  #
                    'learn_rate': 0.001,
                    'drop': 0.5,
                    'fc_size': 64,
                    'blocks': '4_4_3',
                    'l2r': 1e-4,
                    'temperature': 4,
                    'lbl_smt_frac': 0,
                    'n_gpu': 4,
                    'n_cpu': 24,  # 20->16
                    'pw': -8}
    train(train_sample_fn, val_sample_fn, out_model_root_dir, **model_params)




