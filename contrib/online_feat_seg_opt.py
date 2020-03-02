#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: online_feat_seg_opt.py
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


class OnlineFeatSegOpt:

    def __init__(self, sample_id, chr_id, win_size, in_online_whole_chr_feat_fn):

        self.sample_id = sample_id
        self.chr_id = chr_id

        self.__chr_segs_unpredictable = None
        self.__chr_segs_predictable = None
        self.__chr_segs_predictable_feats = None

        self.win_size = win_size

        if not os.path.exists(in_online_whole_chr_feat_fn):
            raise FileNotFoundError('input feature file does not exist. {}'.format(in_online_whole_chr_feat_fn))

        logger.info('loading whole feature map for sample {} chr {} (almost take 2min for chr1)...'.format(
            sample_id, chr_id))
        with np.load(in_online_whole_chr_feat_fn, mmap_mode='r') as f_npz:
            self.whole_chr_feat = f_npz['chr_feat_mat_whole']

    def cal_feat_segs(self, ref_fasta_fn, online_feat_segs_fn,
                      n_features=13, min_r=0.1, stride_size=200):
        """

        :param ref_fasta_fn:
        :param online_feat_segs_fn:
        :param n_features:
        :param min_r:
        :param stride_size:
        :return:
        """

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

        self.__chr_segs_predictable = np.empty((0, 4), dtype=np.int)
        self.__chr_segs_predictable_feats = np.empty((0, self.win_size, n_features))

        seg_gap_inds = np.where(seg_values == 0)[0]
        seg_val_len_less_inds = np.where((seg_values == 1) & (seg_lengths < self.win_size))[0]
        seg_val_normal_inds = np.where((seg_values == 1) & (seg_lengths >= self.win_size))[0]

        assert len(seg_val_len_less_inds) + len(seg_val_normal_inds) + len(seg_gap_inds) == len(seg_values)

        logger.info('calculating gap segments for sample {}, chr {}...'.format(
            self.win_size, stride_size, self.sample_id, self.chr_id))
        self.__chr_segs_unpredictable = np.array([[i_gap_start, i_gap_start + i_gap_lens, i_gap_lens, 0]
                                                  for i_gap_start, i_gap_lens in
                                                  zip(seg_starts[seg_gap_inds], seg_lengths[seg_gap_inds])])

        self.__chr_segs_predictable = np.empty((0, 4), dtype=np.int)
        self.__chr_segs_predictable_feats = np.empty((0, self.win_size, n_features))

        logger.info('calculating {} bp-long {} bp step feature maps sample {}, chr {}...'.format(
            self.win_size, stride_size, self.sample_id, self.chr_id))

        val_seg_len_less_starts = seg_starts[seg_val_len_less_inds]
        val_seg_len_less_lens = seg_lengths[seg_val_len_less_inds]
        val_seg_len_less_end = val_seg_len_less_starts + val_seg_len_less_lens

        val_seg_poss_zip = list(zip(val_seg_len_less_starts, val_seg_len_less_end))

        # slice the seg into win_size
        val_seg_normal_starts = seg_starts[seg_val_normal_inds]
        val_seg_normal_lens = seg_lengths[seg_val_normal_inds]

        val_normal_slice_starts = [seq_slide(i_seg_len, self.win_size, stride_size)
                                   for i_seg_len in val_seg_normal_lens]

        assert len(val_seg_normal_starts) == len(val_normal_slice_starts)
        val_nomarl_slices = [(val_seg_normal_starts[i] + i_slice_start,
                              val_seg_normal_starts[i] + i_slice_start + self.win_size,
                              end_start, remain_len)
                             for i, (i_slice_start, end_start, remain_len) in enumerate(val_normal_slice_starts)]

        for i_seg_norm_starts, i_seg_norm_ends, end_start, remain_len in val_nomarl_slices:
            val_seg_poss_zip.extend(list(zip(i_seg_norm_starts, i_seg_norm_ends)))
            if remain_len > 0:
                val_seg_poss_zip.append((end_start, end_start + remain_len))

        logger.info('saving segments of {} bp-long {} bp step feature maps... {}'.format(
            self.win_size, stride_size, online_feat_segs_fn))

        for i_w_start, i_w_end in val_seg_poss_zip:
            i_w_len = i_w_end - i_w_start
            self.__get_feats_region(i_w_start, i_w_end, i_w_len, min_r)
            if i_w_start < 55000:
                logger.info('process at {}'.format(i_w_start))

        np.savez_compressed(online_feat_segs_fn,
                            chr_segs_unpredictable=self.__chr_segs_unpredictable,
                            chr_segs_predictable=self.__chr_segs_predictable,
                            chr_segs_predictable_feats=self.__chr_segs_predictable_feats)

        logger.info('Done, saving the result file at {}'.format(online_feat_segs_fn))

    def __get_feats_region(self, reg_start, reg_end, reg_len, min_r):
        """

        :param reg_start:
        :param reg_end:
        :param reg_len:
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
                np.array([[reg_start, reg_end, reg_len, 1]], dtype=np.int),
                axis=0)
        elif len(valide_base_cov) < self.win_size * min_r:
            # the number of reads coverage is not enough
            self.__chr_segs_unpredictable = np.append(
                self.__chr_segs_unpredictable,
                np.array([[reg_start, reg_end, reg_len, 2]], dtype=np.int),
                axis=0)
        else:
            if int(reg_len) < self.win_size:
                # ensure the self.win_size length
                i_feat_mat = self.whole_chr_feat[(reg_end-self.win_size):reg_end]
            # ok, requirements satisfied
            # normalize
            i_x_max = np.max(i_feat_mat, axis=0)
            i_x_max[i_x_max == 0] = 1
            i_feat_mat = i_feat_mat * 1.0 / i_x_max.reshape(1, len(i_x_max))

            # padding 0 at end

            self.__chr_segs_predictable = np.append(
                self.__chr_segs_predictable,
                np.array([[reg_start, reg_end, reg_len, 3]], dtype=np.int),
                axis=0)

            self.__chr_segs_predictable_feats = np.append(
                self.__chr_segs_predictable_feats,
                np.array([i_feat_mat]),
                axis=0)