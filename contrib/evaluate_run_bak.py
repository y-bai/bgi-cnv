#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: evaluate_run.py
    Description:
    
Created by Yong Bai on 2019/8/20 3:11 PM.
"""
import os
import numpy as np
from model import cnv_net
import re

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate(model_root_dir, total_samples_ls_fn, test_ratio,
                   win_size, min_size, min_r, min_f_deldup, min_f_neu, is_norm, model_weight_fn):
    
    xtest = []
    ytest = []
    # load test data set
    test_set_fn = os.path.join(model_root_dir,
                               'test{:.2f}_win{}_minsize{}_dataset.npz'.format(test_ratio, win_size, min_size))
    if os.path.exists(test_set_fn):
        with np.load(test_set_fn) as test_set:
            xtest = test_set['x_test']
            ytest = test_set['y_test']
    else:
        from keras.utils import to_categorical
        _, test_ids = get_train_test_ids(model_root_dir, total_samples_ls_fn, test_ratio)
        x_test = []
        y_test_ = []

        for ix, test_id in enumerate(test_ids):
            x, y = get_data(test_id, model_root_dir, win_size, min_r, min_f_deldup, min_f_neu, is_norm)

            if (not (x is None)) and (not (y is None)):
                if len(x_test) == 0:
                    x_test = x
                    y_test_ = y
                else:
                    x_test = np.concatenate((x_test, x), axis=0)
                    y_test_ = np.concatenate((y_test_, y))

        y_ = [1 if x == 'DEL' else 2 if x == 'DUP' else 0 for x in y_test_]
        y_test = to_categorical(y_)
        logger.info('x test: {}, y test: {}'.format(x_test.shape, y_test.shape))
        np.savez_compressed(test_set_fn, x_test=x_test, y_test=y_test)
        xtest = x_test
        ytest = y_test

    n_example, n_win_size, n_feat = xtest.shape
    n_class = ytest.shape[1]
    
    model_weight_name = os.path.splitext(os.path.basename(model_weight_fn))[0]
    model_in_lst = model_weight_name.split('-')
    model_name = model_in_lst[1]
    model_params_lst = re.findall(r"[-+]?\d*\.\d+|\d+", model_in_lst[0])
    logging.info('model name: {0}, model params(batch, epoch, lr, drop, fc, block, win): {1}'.format(
        model_name, model_params_lst))
    
    assert len(model_params_lst) >= 6

    drop = float(model_params_lst[3])
    fc_size = int(model_params_lst[4])
    blocks = (int(x) for x in model_params_lst[5])

    model = None
    if model_name == 'cnvnet':
        model = cnv_net(n_win_size, n_feat, n_class, drop=drop, blocks=blocks, fc_size=fc_size)

    model.load_weights(model_weight_fn)
    logging.info("finished loading model!")
    
    
    ypred = model.predict(xtest)
    test_pred_fn = os.path.join(model_root_dir,
                                'test_out/test{:.2f}_win{}_minsize{}_pred.npz'.format(
                                    test_ratio, win_size, min_size))
    np.savez_compressed(test_pred_fn, ypred=ypred, ytrue=ytest)
    logger.info('Predict Done, result saved at {}'.format(test_pred_fn))
