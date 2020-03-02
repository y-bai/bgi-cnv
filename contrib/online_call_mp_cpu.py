#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: online_call.py
    Description:

Created by Yong Bai on 2019/9/29 10:57 PM.
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
import concurrent.futures
import h5py

import glob
import logging

import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CNVNetwork:

    def __init__(self, weight_path, win_size, n_feat, n_class, drop, _blocks, fc_size, n_proc=10):
        import tensorflow as tf
        from keras import backend as K
        from model import cnv_net

        K.clear_session()
        config = tf.ConfigProto(device_count={"CPU": n_proc},
                                intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(config=config)
        K.set_session(sess)

        model_name = 'cnvnet'
        self.model = cnv_net(win_size, n_feat, n_class, filters=32, kernel_size=16, strides=1, pool_size=2,
                         pool_stride=2, drop=drop, blocks=_blocks, fc_size=fc_size,
                         kernel_regular_l2=None, m_name=model_name)
        self.model.load_weights(weight_path)

    def predict(self, in_data, batch_size=None):

        with h5py.File(in_data, 'r') as i_fn_read:
            i_fn_pred_meta = i_fn_read.get('pred_meta').value
            i_fn_pred_feat = i_fn_read.get('pred_feat').value
        i_feat_arr = np.array(i_fn_pred_feat)
        del i_fn_pred_feat

        ypred = self.model.predict(i_feat_arr, batch_size=batch_size)
        del i_feat_arr
        return i_fn_pred_meta, ypred, in_data


def job_wraper(func_param):
    return pool_job(*func_param)


def pool_job(weight_path, win_size, n_feat, n_class, drop, _blocks, fc_size, in_data, batch_size, n_proc):
    model = CNVNetwork(weight_path, win_size, n_feat, n_class, drop, _blocks, fc_size, n_proc=n_proc)
    return model.predict(in_data, batch_size)


def online_call_mp_cpu(online_seg_data_root_dir, online_call_out_root_dir, model_in_root_dir,
                n_win_size=1000, n_feat=13, n_class=3,
                epochs=64, batch=1024, learn_rate=0.001, drop=0.5,
                fc_size=64, blocks='4_4_3', step_size=200, sample_id='NA12878',
                chr_id='1', min_ratio=0.1, seg_range='a', cost_mat='221', n_proc=10):

    # get model name
    _blocks = tuple(int(x) for x in blocks.split('_'))

    def out_name():
        str_blocks = [str(x) for x in blocks.split('_')]
        str_blk = ''.join(str_blocks)
        return 'b{0}_e{1}_lr{2:.3f}_dr{3:.1f}_fc{4}_blk{5}_win{6}_cw{7}'.format(
            batch, epochs, learn_rate, drop, fc_size, str_blk, n_win_size, cost_mat)

    model_name = 'cnvnet'
    model_weight_fn = os.path.join(model_in_root_dir, out_name() + '-' + model_name + '.hdf5')
    if not os.path.exists(model_weight_fn):
        raise FileNotFoundError('model weight file not found. {}'.format(model_weight_fn))

    # load data
    # default: online_seg_data_root_dir = '/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online'
    online_seg_data_subroot_dir = os.path.join(online_seg_data_root_dir, sample_id + '/data')
    if not os.path.isdir(online_seg_data_subroot_dir):
        raise FileNotFoundError('No segments generated for sample {}, chr {}'.format(sample_id, chr_id))

    part_fname = 'win{0}_step{1}_r{2:.2f}_chr{3}_seg_'.format(n_win_size, step_size, min_ratio, chr_id)
    # create out dir
    # default: online_call_out_root_dir = '/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online'
    online_call_out_subroot_dir = os.path.join(online_call_out_root_dir, sample_id + '/cnv_call')
    if not os.path.isdir(online_call_out_subroot_dir):
        os.mkdir(online_call_out_subroot_dir)
    call_out_fn = os.path.join(online_call_out_subroot_dir,
                               'win{0}_step{1}_r{2:.2f}_chr{3}-cnv-call-'.format(
                                   n_win_size, step_size, min_ratio, chr_id))
    if seg_range != 'a':  # if not whole chr
        part_fname = part_fname + 'p-' + seg_range + '_'
        call_out_fn = call_out_fn + 'p-' + seg_range + '-'
    else:
        part_fname = part_fname + 'a_'
        call_out_fn = call_out_fn + 'a-'

    call_out_fn = call_out_fn + 'result-mp-cpu.csv'

    if os.path.exists(call_out_fn):
        os.remove(call_out_fn)

    gap_h5_fn = os.path.join(online_seg_data_subroot_dir, part_fname+'gap.h5')
    gap_pred = None
    if os.path.exists(gap_h5_fn):
        with h5py.File(gap_h5_fn, 'r') as gap_fn_read:
            gap_obj = gap_fn_read.get('gap')
            if gap_obj:
                gap_result = gap_obj.value
                tmp_arr = np.full((gap_result.shape[0], 4), -1)
                # tmp_arr[:] = np.nan
                gap_pred = np.concatenate((gap_result, tmp_arr), axis=1)
                del tmp_arr

    unpred_h5_fn_list = glob.glob(os.path.join(online_seg_data_subroot_dir, part_fname+'unpred_*'))
    f_unpred_arr = None
    for i_unpred_fn in unpred_h5_fn_list:
        with h5py.File(i_unpred_fn, 'r') as i_uppred_fn_read:
            i_unpred_meta = i_uppred_fn_read.get('unpred_meta').value
        i_unpred_meta_arr = np.array(i_unpred_meta)

        tmp_arr = np.full((i_unpred_meta_arr.shape[0], 4), -1)
        # tmp_arr[:] = np.nan

        if f_unpred_arr is None:
            f_unpred_arr = np.concatenate((i_unpred_meta_arr, tmp_arr), axis=1)
        else:
            unpred_arr = np.concatenate((i_unpred_meta_arr, tmp_arr), axis=1)
            f_unpred_arr = np.concatenate((f_unpred_arr, unpred_arr), axis=0)

    pred_h5_fn_list = glob.glob(os.path.join(online_seg_data_subroot_dir, part_fname+'pred_*'))

    logger.info('calling cnv...')
    f_pred_res = None
    total_fn = len(pred_h5_fn_list)

    qu_tasks = mp.Queue()
    for i_fn in pred_h5_fn_list:
        qu_tasks.put((pool_job, model_weight_fn, n_win_size, n_feat,
                             n_class, drop, _blocks, fc_size, i_fn, batch, n_proc))
    qu_tasks.put(None)

    results = []
    with concurrent.futures.ProcessPoolExecutor(n_proc) as p:
        while True:
            task = qu_tasks.get()
            if task is None:
                break
            func, *args = task
            results.append(p.submit(func, *args))

        i_dex = 0
        for res in results:
            i_dex += 1
            i_pred_meta, i_ypred, i_fn = res.result()
            logger.info('finished {}/{}, {}'.format(i_dex, total_fn, i_fn))
            assert len(i_pred_meta) == i_ypred.shape[0]
            ypred_l = np.argmax(i_ypred, axis=1)

            if f_pred_res is None:
                f_pred_res = np.concatenate((np.array(i_pred_meta), i_ypred, ypred_l[:, np.newaxis]), axis=1)
            else:
                i_pred_res = np.concatenate((np.array(i_pred_meta), i_ypred, ypred_l[:, np.newaxis]), axis=1)
                f_pred_res = np.concatenate((f_pred_res, i_pred_res), axis=0)

    logger.info('combining and sorting results...')
    # check
    if gap_pred is not None and f_unpred_arr is not None and f_pred_res is not None:
        whl_cnv_re = np.concatenate((gap_pred, f_unpred_arr, f_pred_res), axis=0)
    elif gap_pred is not None and f_unpred_arr is not None and f_pred_res is None:
        whl_cnv_re = np.concatenate((gap_pred, f_unpred_arr), axis=0)
    elif gap_pred is not None and f_pred_res is not None and f_unpred_arr is None:
        whl_cnv_re = np.concatenate((gap_pred, f_pred_res), axis=0)
    elif gap_pred is None and f_unpred_arr is not None and f_pred_res is not None:
        whl_cnv_re = np.concatenate((f_unpred_arr, f_pred_res), axis=0)
    elif f_pred_res is None and f_unpred_arr is None and gap_pred is not None:
        whl_cnv_re = gap_pred.copy()
    elif gap_pred is None and f_unpred_arr is None and f_pred_res is not None:
        whl_cnv_re = f_pred_res.copy()
    elif f_pred_res is None and gap_pred is None and f_unpred_arr is not None:
        whl_cnv_re = f_unpred_arr.copy()

    del gap_pred, f_unpred_arr, f_pred_res

    ind = np.argsort(whl_cnv_re[:, 0])
    whl_cnv_re = whl_cnv_re[ind]

    out_df = pd.DataFrame(data=whl_cnv_re,
                          columns=['seg_s', 'seg_e', 'seg_l', 'indicator', 'p_neu', 'p_del', 'p_dup', 'pred_l'])
    out_df[['seg_s', 'seg_e', 'seg_l', 'indicator', 'pred_l']] = out_df[
        ['seg_s', 'seg_e', 'seg_l', 'indicator', 'pred_l']].astype(int)
    out_df.to_csv(call_out_fn, index=False, sep='\t')
    logger.info('Done, online cnv call results saved into {}'.format(call_out_fn))









