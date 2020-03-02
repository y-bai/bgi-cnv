#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: online_cnv_call.py
    Description:
    
Created by Yong Bai on 2019/8/13 2:51 PM.
"""
import sys
import os
import argparse
import numpy as np
import re
import logging

sys.path.append("..")
from cnv_utils import FeatureOnline
from model import cnv_net

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(args):
    sample_id = args.sample_id
    in_bam_fn = args.in_bam_fn
    min_ratio = args.min_ratio
    win_size = args.win_size
    chr_id = args.chr_id
    ref_fa_f = args.ref_fa_f
    ref_map_f = args.ref_map_f
    step_size = args.step_size

    model_weight_fn = args.model_weight_fn
    out_dir = args.out_dir

    n_features = args.n_features
    n_classes = args.n_classes

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
        model = cnv_net(win_size, n_features, n_classes, drop=drop, blocks=blocks, fc_size=fc_size)

    model.load_weights(model_weight_fn)
    logging.info("finished loading model!")

    out_fn = os.path.join(out_dir, model_weight_name + '_' + sample_id + '.online_cnv_call')
    if os.path.exists(out_fn):
        os.remove(out_fn)

    # generate online features
    online_feat_obj = FeatureOnline(ref_fa_f, ref_map_f)
    online_feat_obj.load_bam(in_bam_fn)

    def online_call(tmp_online_feats, tmp_chr_id):
        for i, res in enumerate(tmp_online_feats):
            reg_start, reg_end, reg_len, out_indicator, f_mat = res
            if out_indicator == 3:
                # normalization
                i_x_max = np.max(f_mat, axis=-1)
                i_x_max[i_x_max == 0] = 1
                f_mat = f_mat * 1.0 / i_x_max.reshape(n_features, 1)
                f_mat = np.transpose(f_mat)
                y_prob = model.predict(np.array([f_mat]), batch_size=1)[0]  # batch_size=1?
                ypred_prob = ','.join(['{:.6f}'.format(x) for x in y_prob])
                ypred = y_prob.argmax(axis=-1)
                with open(out_fn, 'a') as f:
                    f.write(
                        '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(tmp_chr_id, reg_start, reg_end, reg_len,
                                                              out_indicator, ypred_prob, ypred))

            else:
                # save to the result out put
                with open(out_fn, 'a') as f:
                    f.write(
                        '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(tmp_chr_id, reg_start, reg_end, reg_len,
                                                              out_indicator, None, None))
        del tmp_online_feats
        return 'chromosome {} cnv call done!'.format(tmp_chr_id)

    logging.info('begining calling cnvs for chromosome: {}'.format(chr_id))
    if chr_id.upper() == 'A':  # all chromosomes
        chr_lst_idx = [str(i) for i in range(1, 23)]  # + ['X']
        for i_chr_id in chr_lst_idx:
            online_feats = online_feat_obj.run(sample_id, i_chr_id, win_size=win_size,
                                               min_r=min_ratio, stride_size=step_size)
            logging.info(online_call(online_feats, i_chr_id))
    else:
        online_feats = online_feat_obj.run(sample_id, chr_id, win_size=win_size,
                                           min_r=min_ratio, stride_size=step_size)
        logging.info(online_call(online_feats, chr_id))

    logging.info('Sample {} cnv call completed, output: {}'.format(sample_id, out_fn))


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
        "-d",
        "--model_weight_fn",
        type=str,
        help="deep neural network weight file name")

    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="predict output directory")

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
        "-p",
        "--step_size",
        type=int,
        default=1000,
        help="step size when sliding window")

    # /zfssz3/SOLEXA_DATA/BC_RD_P2/PROJECT/F16ZQSI1SH2595/PMO/140K.bam.list

    parser.add_argument("-f", '--ref_fa_f', type=str, help='reference fasta file')
    parser.add_argument("-m", '--ref_map_f', type=str, help='reference mappability bw file')

    args = parser.parse_args()
    main(args)
