#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: model_crt_dataset4generator2.py
    Description:
    same the training data sets by h5
    
Created by Yong Bai on 2019/9/17 11:28 AM.
"""
import os
import numpy as np
import pandas as pd
import argparse
import logging
import multiprocessing as mp
# from tempfile import NamedTemporaryFile
import pickle
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

import sys
sys.path.append("..")
from cnv_utils import str2bool


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def wrap_cvt_dataset_one_sample(in_args):
    return cvt_dataset_one_sample(*in_args)


def cvt_dataset_one_sample(fn, ntf_path):

    """

    :param fn:
    :param sample_id:
    :param out_dataset_dir:
    :param m_feats:
    :param is_norm:
    :return:
    """
    x_fn = fn + '.x'
    y_fn = fn + '.y'
    # 'ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.{0}.cnvs.training.{1}_{2:.2f}_{3:.2f}_{4}'
    fn_basename = os.path.basename(fn)
    fn_split = fn_basename.split('_')
    head_strs = fn_split[0].split('.')
    sample_id = head_strs[9]
    is_norm = str2bool(fn_split[3])
    w_size = int(head_strs[12])

    # xs = np.loadtxt(x_fn)  # np.loadtxt() is too slow
    try:
        xs = pd.read_csv(x_fn, sep='\s+', header=None).values
    except Exception as ex:
        print('========================================================\n'
              'Error: {0}. \n{1}'.format(ex, x_fn))
        return 'NOT_FILE'

    with open(y_fn, 'r') as y_f:
        ys = y_f.read().splitlines()

    # 13 features
    m_feats = 13
    n_xs = int(len(xs) / m_feats)

    if n_xs != len(ys):
        logger.error('***************************************************\n'
                     'Error, the number of feature map n={0} not equal to the number of labels n={1}. '
                     '\n for details, see {2} and {3}'.format(n_xs, len(ys), x_fn, y_fn))
        os._exit(1)

    logger.info('finished loading feature map file {0}, shape={1}, n_samples={2}, '.format(x_fn, xs.shape, n_xs))
    npx = np.zeros((n_xs, w_size, m_feats))
    label_y = []

    for i in range(n_xs):
        i_x = xs[i*m_feats:(i+1) * m_feats, :]

        # normalize
        if not is_norm:
            i_x_max = np.max(i_x, axis=1)
            i_x_max[i_x_max == 0] = 1
            i_x = i_x * 1.0 / i_x_max.reshape(m_feats, 1)
        if i_x.shape[1] != w_size:
            logger.error('***************************************************\n'
                         'Error, the len of feature map ({0})not equal to w_size={1}. '
                         '\n for details, see {2}'.format(i_x.shape[1], w_size, x_fn))
            os._exit(0)

        npx[i] = np.transpose(i_x)

        label_y.append(sample_id + '|' + ys[i])

    out_dict = {'f_map': npx, 'y_str': label_y, 'y_len': len(label_y)}
    with open(ntf_path, 'wb') as ntf_f:
        pickle.dump(out_dict, ntf_f)

    return ntf_path


def cal_len_dataset(y_fn):
    with open(y_fn+'.y', 'r') as y_f:
        ys = y_f.read().splitlines()
    return len(ys)


def main(args):
    # mp.set_start_method('forkserver')

    train_test_fn = args.train_test_fn
    win_size = args.win_size
    min_r = args.ratio
    cnv_min_f = args.cnv_min_f
    neu_min_f = args.neu_min_f
    is_norm = args.normalize
    n_cpu = args.n_cpu

    out_dir_root = args.out_dir
    in_cnv_data_dir = args.in_cnv_data_dir
    in_neu_data_dir = args.in_neu_data_dir

    n_features = 13

    in_cnv_win_dir = os.path.join(in_cnv_data_dir, '{}_ds_trains'.format(win_size))
    in_neu_win_dir = os.path.join(in_neu_data_dir, '{}_ds_neus_trains'.format(win_size))

    out_data_dir = os.path.join(out_dir_root, 'win_{}'.format(win_size))
    if not os.path.isdir(out_data_dir):
        os.mkdir(out_data_dir)

    logger.info('loading train sample ids and test ids...')
    with np.load(train_test_fn) as sample_ids:
            train_ids, test_ids = sample_ids['train_ids'], sample_ids['test_ids']

    y_fn_fmt = 'ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.{0}.cnvs.training.{1}_{2:.2f}_{3:.2f}_{4}'
    f_cnv = lambda x: os.path.join(in_cnv_win_dir, y_fn_fmt.format(x, win_size, min_r, cnv_min_f, is_norm))
    f_neu = lambda x: os.path.join(in_neu_win_dir, y_fn_fmt.format(x, win_size, min_r, neu_min_f, is_norm))
    trains_fn = [f(x) for x in train_ids for f in (f_cnv, f_neu)]
    tests_fn = [f(x) for x in test_ids for f in (f_cnv, f_neu)]

    # trains_fn = trains_fn1[0:4]
    # tests_fn = tests_fn1[0:4]

    n_trains = 0
    n_tests = 0
    for i_train_fn in trains_fn:
        n_trains += cal_len_dataset(i_train_fn)
    for i_test_fn in tests_fn:
        n_tests += cal_len_dataset(i_test_fn)

    logger.info('the number of {}-long samples: n_trains={}, n_tests={}'.format(win_size, n_trains, n_tests))

    # create h5 files for saving the datasets
    fn_train_out_fn = os.path.join(out_data_dir,
                                   'trains_mr{0:.2f}_mf{1:.2f}_original.h5'.format(min_r, cnv_min_f))
    if os.path.exists(fn_train_out_fn):
        os.remove(fn_train_out_fn)
    fn_test_out_fn = os.path.join(out_data_dir,
                                  'tests_mr{0:.2f}_mf{1:.2f}_original.h5'.format(min_r, cnv_min_f))
    if os.path.exists(fn_test_out_fn):
        os.remove(fn_test_out_fn)

    chk_size = 8
    h5_str_dt = h5py.special_dtype(vlen=str)

    # h5 file for train
    logger.info('deal with training dataset...')
    h5f_trains = h5py.File(fn_train_out_fn, 'w')
    train_x_dset = h5f_trains.create_dataset('x',
                                           (n_trains, win_size, n_features),
                                           maxshape=(None, win_size, n_features),
                                           dtype=np.float32,
                                           chunks=(chk_size, win_size, n_features),  # 512 * 8->256 for 100k win
                                           compression="gzip", compression_opts=4)
    train_x_dset.attrs['n_sample'] = n_trains
    train_y_dset = h5f_trains.create_dataset('str_y',
                                             (n_trains,),
                                             maxshape=(None, ),
                                             dtype=h5_str_dt,
                                             chunks=(chk_size, ),
                                             compression="gzip", compression_opts=4)

    # if data is extremely huge, then we will get
    # Reason: 'error("'i' format requires -2147483648 <= number <= 2147483647",)'
    # this is python 3.6 bug
    # thus we use NamedTemporaryFile
    # trains_tmp_ntf = [NamedTemporaryFile(delete=False, prefix=os.path.basename(x)+'_') for x in trains_fn]
    # trains_tmp_ntf_fn = [x.name for x in trains_tmp_ntf]

    tmp_dir = '/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/tmp'
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)
    trains_tmp_ntf_fn = [os.path.join(tmp_dir, os.path.basename(x) + '_tr_' + str(i)) for i, x in enumerate(trains_fn)]

    # for i_f in trains_tmp_ntf_fn:
    #     if os.path.exists(i_f):
    #         os.remove(i_f)
    #     try:
    #         i_f.close()
    #     except Exception as inst:
    #         logger.info(inst)
    #         continue

    train_fn_ntf = list(zip(trains_fn, trains_tmp_ntf_fn))

    train_p = mp.Pool(n_cpu)
    trains_results = train_p.imap(wrap_cvt_dataset_one_sample, train_fn_ntf)
    logger.info('saving the training dataset...')
    i_start = 0
    for i, i_ntf in enumerate(trains_results):
        if i_ntf == 'NOT_FILE':
            # win = 100000 has file not found, need to check
            logger.error('>>>>>>>>>>>>>>>>Error found, please check corresponding output file')
            continue
        with open(i_ntf, 'rb') as i_ntf_f:
            i_out_dict = pickle.load(i_ntf_f)
        len_y = i_out_dict['y_len']
        train_x_dset[i_start: i_start+len_y] = i_out_dict['f_map']
        train_y_dset[i_start: i_start+len_y] = i_out_dict['y_str']
        i_start = i_start + len_y
        os.unlink(i_ntf)
        logger.info('whole train data generation finished, {}/{}'.format(i+1, len(trains_fn)))
    h5f_trains.close()
    train_p.close()
    train_p.join()
    ###########################################################################
    logger.info('deal with test dataset...')
    # h5 file for test
    h5f_tests = h5py.File(fn_test_out_fn, 'w')
    test_x_dset = h5f_tests.create_dataset('x',
                                           (n_tests, win_size, n_features),
                                           maxshape=(None, win_size, n_features),
                                           dtype=np.float32,
                                           chunks=(chk_size, win_size, n_features),  # 512 * 8
                                           compression="gzip", compression_opts=4)
    test_x_dset.attrs['n_sample'] = n_tests
    test_y_dset = h5f_tests.create_dataset('str_y',
                                           (n_tests,),
                                           maxshape=(None,),
                                           dtype=h5_str_dt,
                                           chunks=(chk_size,),
                                           compression="gzip", compression_opts=4)

    # tests_tmp_ntf = [NamedTemporaryFile(delete=False, prefix=os.path.basename(x) + '_') for x in tests_fn]
    # tests_tmp_ntf_fn = [x.name for x in tests_tmp_ntf]
    tests_tmp_ntf_fn = [os.path.join(tmp_dir, os.path.basename(x) + '_te_' + str(i)) for i, x in enumerate(tests_fn)]

    # for i_f in tests_tmp_ntf_fn:
    #     if os.path.exists(i_f):
    #         os.remove(i_f)
    #     try:
    #         i_f.close()
    #     except Exception as inst:
    #         logger.info(inst)
    #         continue

    test_fn_ntf = list(zip(tests_fn, tests_tmp_ntf_fn))
    test_p = mp.Pool(n_cpu)
    tests_results = test_p.imap(wrap_cvt_dataset_one_sample, test_fn_ntf)

    logger.info('saving the test dataset...')
    i_start = 0
    for i, i_ntf in enumerate(tests_results):
        if i_ntf == 'NOT_FILE':
            logger.error('>>>>>>>>>>>>>>>>Error found, please check corresponding output file')
            continue
        with open(i_ntf, 'rb') as i_ntf_f:
            i_out_dict = pickle.load(i_ntf_f)
        len_y = i_out_dict['y_len']
        test_x_dset[i_start: i_start + len_y] = i_out_dict['f_map']
        test_y_dset[i_start: i_start + len_y] = i_out_dict['y_str']
        i_start = i_start + len_y
        os.unlink(i_ntf)
        logger.info('whole test data generation finished, {}/{}'.format(i + 1, len(tests_fn)))

    h5f_tests.close()
    test_p.close()
    test_p.join()

    logger.info('Done, data save at {}'.format(out_data_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create data sample list ')
    parser.add_argument(
        "-f",
        "--train_test_fn",
        type=str,
        help="name of file containing training and testing sample ids",
        required=True)

    parser.add_argument(
        "-w",
        "--win_size",
        type=int,
        default=1000,
        help='window size. This will be hyperparamter')

    parser.add_argument(
        "-r",
        "--ratio",
        type=float,
        default=0.1,
        help='cnv region that has read coverage less than the ratio will be filtered out. This will be hyperparameter')

    parser.add_argument(
        "-q",
        "--cnv_min_f",
        type=float,
        default=0.01,
        help='cnv whose frequency less than the frequency will be filtered out. This will be hyperparameter')

    parser.add_argument(
        "-p",
        "--neu_min_f",
        type=float,
        default=-2,
        help='neu whose frequency less than the frequency will be filtered out. This will be hyperparameter')

    parser.add_argument(
        "-n",
        "--normalize",
        type=str2bool,
        default=False,
        help='if normalizing the training data, if input is .x,.y,.f files, then the data is not normalized')

    parser.add_argument(
        "-v",
        "--in_cnv_data_dir",
        type=str,
        help="input dir where storing text feature files for cnvs(ie, .x,.y,.f files)")

    parser.add_argument(
        "-u",
        "--in_neu_data_dir",
        type=str,
        help="input dir where storing text feature files for neus(ie, .x,.y,.f files)")
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="output directory",
        required=True)

    parser.add_argument(
        "-t",
        "--n_cpu",
        type=int,
        default=4,
        help="the number of cpus")

    args = parser.parse_args()
    logger.info('args: {}'.format(args))
    main(args)
