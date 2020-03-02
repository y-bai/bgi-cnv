#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: model_train_main.py
    Description:
    
Created by Yong Bai on 2019/8/20 3:13 PM.
"""
from train_model import (cv_train, load_train_data, mnd_train, my_generator,
                         train)

if __name__ == "__main__":

    # path for save the data
    model_root_dir = "/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/npz_feature_file/winsize_1000"
    total_samples_ls_fn = "/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/sample.list"

    
    min_r = 0.1
    min_f_deldup = 0.01
    min_f_neu = -2
    is_norm = False
    
    win_size = 1000
    batch_size=256
    test_ratio = 0.1
    validation_ratio=0.1

    example_min_size = 160000

    # x_train, y_train = load_train_data(model_root_dir, total_samples_ls_fn,
                                    #    win_size, min_r, min_f_deldup, min_f_neu, is_norm, test_ratio, example_min_size)

    # X, y=my_generator(model_root_dir, total_samples_ls_fn,
                                    #    win_size, min_r, min_f_deldup, min_f_neu, is_norm, test_ratio, example_min_size)
    train_generator,validation_generator = my_generator(model_root_dir, total_samples_ls_fn, win_size, min_r, min_f_deldup, min_f_neu, is_norm,
                                                         test_ratio=test_ratio, validation_ratio=validation_ratio, batch_size=batch_size, n_classes = 3, shuffle='Local_shuffle')
    # print(len(X))
    # multi gpu train
    train(train_generator,validation_generator, model_root_dir)
    # cv_train(x_train, y_train, model_root_dir)
    # mnd_train(x_train, y_train, model_root_dir)
