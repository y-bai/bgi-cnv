#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: online_cnv_call_AE_main.py
    Description:
    
Created by Yong Bai on 2019/8/20 3:13 PM.
"""
import os
import logging
import argparse
# from online import online_call
from online import online_call_mp_gpu_ae
# from online import online_call_mp_cpu

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(args):

    win_size = args.win_size
    min_r = args.min_ratio
    nb_cls_prop = args.model_type
    n_feat = args.n_features
    n_class = args.n_classes
    sample_id = args.sample_id
    chr_id = args.chr_id
    step_size = args.step_size
    epochs = args.epochs
    batch_size = args.batch_size
    drop_out = args.drop_out
    lr = args.learn_rate
    fc_size = args.fc_size
    blocks = args.blocks
    n_proc = args.n_proc

    pw = args.penality_index
    temperature = args.temperature
    l2r = args.l2r
    label_smooth_factor = args.label_smooth_factor

    model_root_dir = args.model_root_dir
    in_segs_root_dir = args.input_segs_root_dir
    cnv_out_root_dir = args.out_dir

    n_seg_range = args.n_seg_range

    n_res_blks = -1
    t_win_size = win_size
    while t_win_size % 2 == 0:
        n_res_blks += 1
        t_win_size = int(t_win_size // 2)

    if n_res_blks > 0:
        f_win_size = win_size // (2 ** (n_res_blks + 1))
    else:
        f_win_size = win_size

    params = {'n_win_size': f_win_size,
              'orig_win': win_size,
              'n_feat': n_feat,
              'n_class': n_class,
              'epochs': epochs,
              'batch': batch_size,
              'learn_rate': lr,
              'drop': drop_out,
              'fc_size': fc_size,
              'blocks': blocks,
              'step_size': step_size,
              'sample_id': sample_id,
              'chr_id': chr_id,
              'min_ratio': min_r,
              'seg_range': n_seg_range,
              'l2r': l2r,
              'temperature': temperature,
              'lbl_smt_frac': label_smooth_factor,
              'filters': 32,
              'kernel_size': 16,
              'pw': pw,
              'n_proc': n_proc}

    logger.info('params: {}'.format(params))

    model_sub_root_dir = os.path.join(model_root_dir, 'model{}'.format(''.join(nb_cls_prop.split(':'))))
    model_weight_dir = os.path.join(model_sub_root_dir, 'model_weight')

    online_call_mp_gpu_ae(in_segs_root_dir, cnv_out_root_dir, model_weight_dir, **params)
    # online_call_mp_cpu(in_segs_root_dir, cnv_out_root_dir, model_weight_dir, **params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='online cnv call')

    parser.add_argument(
        "-s",
        "--sample_id",
        type=str,
        help="input sample id",
        required=True)

    parser.add_argument(
        "-c",
        "--chr_id",
        type=str,
        help="chromosome id",
        default='1')

    parser.add_argument(
        "-r",
        '--min_ratio',
        type=float,
        default=0.1,
        help="cnv region that has read coverage less than the ratio will be filtered out. This will be hyperparameter")

    parser.add_argument(
        "-w",
        "--win_size",
        type=int,
        default=1000,
        help='window size. This will be hyperparamter')

    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default='/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online',
        help="predict output directory")

    parser.add_argument(
        "-i",
        "--input_segs_root_dir",
        type=str,
        default='/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online',
        help="segments input directory")

    parser.add_argument(
        "-g",
        "--model_root_dir",
        type=str,
        default='/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out',
        help="model weight root directory")

    parser.add_argument(
        "-n",
        "--n_features",
        type=int,
        default=13,
        help="number of features")

    parser.add_argument(
        "-t",
        "--n_classes",
        type=int,
        default=3,
        help="number of the output classes")

    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default='1:1:1',
        help="model type")

    parser.add_argument(
        "-p",
        "--step_size",
        type=int,
        default=200,
        help="step size when sliding window")

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=64,
        help="model epoches when train")

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1024,
        help="batch size when training model")

    parser.add_argument(
        "-d",
        "--drop_out",
        type=float,
        default=0.5,
        help="drop out ratio when training model")

    parser.add_argument(
        "-l",
        "--learn_rate",
        type=float,
        default=0.001,
        help="lr when training model")

    parser.add_argument(
        "-f",
        "--fc_size",
        type=int,
        default=64,
        help="fully connection size when training model")

    parser.add_argument(
        "-k",
        "--blocks",
        type=str,
        default='4_4_3',
        help="model block structure")

    parser.add_argument(
        "-x",
        "--n_seg_range",
        type=str,
        default='a',
        help="segment range, 'a' for whole chr, otherwise seg_s-seg_e, eg, 10000-30000")

    parser.add_argument(
        "-u",
        "--penality_index",
        type=int,
        default=1,
        help="cost matrices indicator")

    parser.add_argument(
        "-z",
        "--n_proc",
        type=int,
        default=4,
        help='number of multi_processor')

    parser.add_argument(
        "-a",
        "--temperature",
        type=int,
        default=5,
        help='temperature to calibrate the softmax result')

    parser.add_argument(
        "-y",
        "--l2r",
        type=float,
        default=0.0001,
        help='network l2 regularization coff')

    parser.add_argument(
        "-q",
        "--label_smooth_factor",
        type=float,
        default=0.1,
        help='label smoothing factor')

    args = parser.parse_args()
    logger.info(args)
    main(args)