#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: online_feat_seg.py
    Description:
    
Created by Yong Bai on 2019/9/23 12:52 PM.
"""

import os
import sys
import numpy as np
import pandas as pd

import pysam
import logging

sys.path.append("..")
# from online_feat_gen.large_file_reader import CachedLineList
from cnv_utils.feat_gen_utils import load_gap
from cnv_utils.general_utils import find_seg, seq_slide

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OnlineFeatSeg:

    def __init__(self, sample_id, chr_id, in_online_whole_chr_feat_fn):

        self.sample_id = sample_id
        self.chr_id = chr_id

        self.__chr_segs_unpredictable = None
        self.__chr_segs_predictable = None
        self.__chr_segs_predictable_feats = None

        if not os.path.exists(in_online_whole_chr_feat_fn):
            raise FileNotFoundError('input feature file does not exist. {}'.format(in_online_whole_chr_feat_fn))

        logger.info('loading whole feature map for sample {} chr {}...'.format(sample_id, chr_id))
        with np.load(in_online_whole_chr_feat_fn, mmap_mode='r') as f_npz:
            self.whole_chr_feat = f_npz['chr_feat_mat_whole']

    def cal_feat_segs(self, ref_fasta_fn, online_feat_segs_fn,
                      win_size=1000, n_features=13, min_r=0.1, stride_size=200):

        if not os.path.exists(ref_fasta_fn):
            raise FileNotFoundError('Reference fasta file does not exist. {}'.format(ref_fasta_fn))

        logger.info('loading Reference fasta file...')
        ref_fa_obj = pysam.FastaFile(ref_fasta_fn)

        # reference base
        logger.info('Loading reference sequence for sample {} chr: {}...'.format(self.sample_id, self.chr_id))
        rb_base_chr = ref_fa_obj.fetch('chr' + self.chr_id)
        chr_len = len(rb_base_chr)

        ref_gap_obj = load_gap()
        ref_gap_chr = ref_gap_obj.loc[ref_gap_obj['CHROM'] == 'chr' + self.chr_id,
                                      ['START', 'END']] if not ref_gap_obj.empty else None

        fil_pos = np.array([], dtype=np.int)
        if ref_gap_chr is not None:
            for _, i_row in ref_gap_chr.iterrows():
                # END is excluded
                fil_pos = np.concatenate((fil_pos, np.arange(i_row['START'], i_row['END'])))

        rb_base_pos = np.ones(chr_len, dtype=int)
        rb_base_pos[fil_pos] = 0

        seg_values, seg_starts, seg_lengths = find_seg(rb_base_pos)
        assert len(seg_values) == len(seg_starts) == len(seg_lengths)

        logger.info('segmenting {} bp-long {} bp step feature maps sample {}, chr {}...'.format(
            win_size, stride_size, self.sample_id, self.chr_id))
        logger.info('>>>>>>>>>>this processing will take a few minutes (almost 180 minutes for chr 1)...')

        self.__chr_segs_unpredictable = np.empty((0, 4), dtype=np.int)
        self.__chr_segs_predictable = np.empty((0, 4), dtype=np.int)
        self.__chr_segs_predictable_feats = np.empty((0, win_size, n_features))

        for val_idx, i_val in enumerate(seg_values):
            if i_val == 0:
                # gap region
                self.__chr_segs_unpredictable = np.append(
                    self.__chr_segs_unpredictable,
                    np.array([[seg_starts[val_idx],
                               seg_starts[val_idx] + seg_lengths[val_idx],
                               seg_lengths[val_idx], 0]], dtype=np.int),
                    axis=0)

            else:
                i_seg_start = seg_starts[val_idx]
                i_seg_len = seg_lengths[val_idx]

                if i_seg_len >= win_size:
                    i_start_indices, remain_len = seq_slide(i_seg_len, win_size, stride_size)
                    for i in i_start_indices:
                        i_w_start = int(i+i_seg_start)
                        i_w_end = int(i_w_start + win_size)

                        logger.info('processing at {}'.format(i_w_start))

                        self.__get_feats_region(i_w_start, i_w_end, win_size, min_r)

                    if remain_len > 0:
                        i_w_start = int(i_seg_len-win_size)
                        i_w_end = int(i_seg_len)

                        self.__get_feats_region(i_w_start, i_w_end, win_size, min_r)
                else:
                    self.__get_feats_region(i_seg_start, i_seg_start+i_seg_len, win_size, min_r)

        logger.info('saving segments of {} bp-long {} bp step feature maps... {}'.format(
            win_size, stride_size, online_feat_segs_fn))
        np.savez_compressed(online_feat_segs_fn,
                            chr_segs_unpredictable=self.__chr_segs_unpredictable,
                            chr_segs_predictable=self.__chr_segs_predictable,
                            chr_segs_predictable_feats=self.__chr_segs_predictable_feats)

        logger.info('Done, saving the result file at {}'.format(online_feat_segs_fn))

    def __get_feats_region(self, reg_start, reg_end, win_size, min_r):
        """

        :param reg_start:
        :param reg_end:
        :param win_size:
        :param min_r:
        :return:
        """

        i_feat_mat = self.whole_chr_feat[reg_start:reg_end]

        i_base_cov = i_feat_mat[:, 0]
        valide_base_cov = i_base_cov[i_base_cov > 0]
        if len(valide_base_cov) == 0:
            # no reads cover
            self.__chr_segs_unpredictable = np.append(
                self.__chr_segs_unpredictable,
                np.array([[reg_start, reg_end, win_size, 1]], dtype=np.int),
                axis=0)
        elif len(valide_base_cov) < win_size * min_r:
            # the number of reads coverage is not enough
            self.__chr_segs_unpredictable = np.append(
                self.__chr_segs_unpredictable,
                np.array([[reg_start, reg_end, win_size, 2]], dtype=np.int),
                axis=0)
        else:
            # ok, requirements satisfied
            # normalize
            i_x_max = np.max(i_feat_mat, axis=0)
            i_x_max[i_x_max == 0] = 1
            i_feat_mat = i_feat_mat * 1.0 / i_x_max.reshape(1, len(i_x_max))

            self.__chr_segs_predictable = np.append(
                self.__chr_segs_predictable,
                np.array([[reg_start, reg_end, win_size, 3]], dtype=np.int),
                axis=0)

            self.__chr_segs_predictable_feats = np.append(
                self.__chr_segs_predictable_feats,
                np.array([i_feat_mat]),
                axis=0)