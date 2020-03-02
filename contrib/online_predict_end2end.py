#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: online_predict_end2end.py
    Description: online cnv call end to end
    
Created by Yong Bai on 2019/9/19 2:39 PM.
"""
import os
import argparse
import logging
import sys

sys.path.append("..")
from online import OnlineCNVCall

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(args):

    sample_id = args.sample_id
    in_bam_fn = args.in_bam_fn

    min_ratio = args.min_ratio
    win_size = args.win_size
    chr_id = args.chr_id.split(',')
    ref_fa_f = args.ref_fa_f
    ref_map_f = args.ref_map_f
    step_size = args.step_size

    online_out_root_dir = args.online_out_root_dir

    model_weight_fn = args.model_weight_fn
    n_features, n_classes = args.n_features, args.n_classes

    n_proc = args.n_proc

    online_feat_obj = OnlineCNVCall(model_weight_fn, n_features, n_classes)
    online_feat_obj.load_deps(in_bam_fn, ref_fa_f, ref_map_f, fmt='bam')

    online_out_sample_subdir = os.path.join(online_out_root_dir, sample_id)
    if not os.path.isdir(online_out_sample_subdir):
        os.mkdir(online_out_sample_subdir)

    online_out_sample_data_dir = os.path.join(online_out_sample_subdir, 'data')
    if not os.path.isdir(online_out_sample_data_dir):
        os.mkdir(online_out_sample_data_dir)
    online_out_cnv_dir = os.path.join(online_out_sample_subdir, 'cnv_call')
    if not os.path.isdir(online_out_cnv_dir):
        os.mkdir(online_out_cnv_dir)

    for i_chr_id in chr_id:
        i_feat_seg_fn = os.path.join(online_out_sample_data_dir,
                                     'win{0}_step{1}_r{2:.2f}_chr{3}_feat_segs.npz'.format(win_size,
                                                                                           step_size,
                                                                                           min_ratio,
                                                                                           i_chr_id))

        i_out_cnv_call_fn = os.path.join(online_out_cnv_dir,
                                         'win{0}_step{1}_r{2:.2f}_chr{3}_cnv_call_res.csv'.format(win_size,
                                                                                                  step_size,
                                                                                                  min_ratio,
                                                                                                  i_chr_id))
        online_feat_obj.cal_feat_segs(sample_id, i_chr_id,
                                      win_size=win_size,
                                      min_r=min_ratio,
                                      stride_size=step_size,
                                      online_feat_segs_fn=i_feat_seg_fn, n_proc=n_proc)

        online_feat_obj.cnv_call(i_out_cnv_call_fn, online_feat_segs_in_fn=i_feat_seg_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate cnv labels')
    parser.add_argument(
        "-b",
        "--in_bam_fn",
        type=str,
        help="input bam file",
        required=True)

    parser.add_argument(
        "-s",
        "--sample_id",
        type=str,
        help="input sample id",
        required=True)

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
        "-c",
        "--chr_id",
        type=str,
        default='A',
        help="chromosome id, 'A' means all chromosomes, otherwise, provide like '1'")

    parser.add_argument(
        "-o",
        "--online_out_root_dir",
        type=str,
        default='/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online',
        help="online output directory")

    parser.add_argument(
        "-p",
        "--step_size",
        type=int,
        default=200,
        help="step size when sliding window")

    parser.add_argument(
        "-d",
        "--model_weight_fn",
        type=str,
        help="model weight file name")

    parser.add_argument(
        "-u",
        "--n_features",
        type=int,
        default=13,
        help="the number of features")

    parser.add_argument(
        "-l",
        "--n_classes",
        type=int,
        default=3,
        help="the number of classes")

    parser.add_argument(
        "-t",
        "--n_proc",
        type=int,
        default=16,
        help="the number of processor")

    # /zfssz3/SOLEXA_DATA/BC_RD_P2/PROJECT/F16ZQSI1SH2595/PMO/140K.bam.list

    parser.add_argument("-f", '--ref_fa_f', type=str, help='reference fasta file')
    parser.add_argument("-m", '--ref_map_f', type=str, help='reference mappability bw file')

    args = parser.parse_args()
    main(args)
