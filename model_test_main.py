#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: model_train_main.py
    Description:
    
Created by Yong Bai on 2019/8/20 3:13 PM.
"""
import os
from train_model import evaluate


if __name__ == "__main__":

    # path for save the data: old path
    # model_root_dir = "/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/1dcnn_resnet"
    # total_samples_ls_fn = "/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/sample.list"
    # model_weight_fn = '/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/1dcnn_resnet/final_model/model_weight/b256_e50_lr0.001_dr0.5_fc128_blk443-cnvnet.hdf5'

    # path for save the test results
    out_root_dir = "/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out"
    # path of test samples
    in_data_root_dir = '/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data'

    win_size = 1000
    min_r = 0.1
    min_f = 0.01
    nb_cls_prop = '1:1:1'

    class_weights = {0: 1, 1: 2, 2: 2}  # this is only used as identification of cost

    test_sample_fn = os.path.join(in_data_root_dir,
                                  'w{0}_r{1:.2f}_f{2:.2f}_test.csv'.format(win_size, min_r, min_f))

    out_model_root_dir = os.path.join(out_root_dir, 'model{}'.format(''.join(nb_cls_prop.split(':'))))

    model_weight_dir = os.path.join(out_model_root_dir, 'model_weight')

    # batch_size = 512
    # model_params = {'n_win_size': win_size,
    #                 'n_feat': 13,
    #                 'n_class': 3,
    #                 'epochs': 50,
    #                 'batch': 512,
    #                 'learn_rate': 0.001,
    #                 'drop': 0.1,
    #                 'fc_size': 32,
    #                 'blocks': '4_4_1',
    #                 'n_cpu': 36,  #
    #                 'class_weights': class_weights}
    #
    # evaluate(test_sample_fn, out_model_root_dir, model_weight_dir, **model_params)

    # batch_size = 1024
    model_params = {'n_win_size': win_size,
                    'n_feat': 13,
                    'n_class': 3,
                    'epochs': 64,
                    'batch': 512, # manually change from 1024 to 512
                    'learn_rate': 0.001,
                    'drop': 0.5,
                    'fc_size': 64,
                    'blocks': '4_4_3',
                    'n_cpu': 36,
                    'class_weights': class_weights}
    evaluate(test_sample_fn, out_model_root_dir, model_weight_dir, **model_params)