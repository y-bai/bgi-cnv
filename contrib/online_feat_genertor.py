#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: online_feat_genertor.py
    Description:
    
Created by Yong Bai on 2019/8/14 2:13 PM.
"""
import os
import numpy as np

import pysam
import pyBigWig
import logging

from cnv_utils.feat_gen_utils import load_gap, gen_feat_region
from cnv_utils.general_utils import find_seg, seq_slide

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureOnline:

    def __init__(self, ref_fasta_fn, ref_map_fn):
        """

        :param ref_fasta_fn:
        :param ref_map_fn:
        :param bam_fn:
        """

        if not os.path.exists(ref_fasta_fn):
            raise FileNotFoundError('Reference fasta file does not exist. {}'.format(ref_fasta_fn))
        if not os.path.exists(ref_map_fn):
            raise FileNotFoundError('Reference mappability bw file does not exist. {}'.format(ref_map_fn))

        logger.info('loading Reference fasta file...')
        ref_fa_obj = pysam.FastaFile(ref_fasta_fn)
        logger.info('loading Reference mappability bw file...')
        ref_bw_obj = pyBigWig.open(ref_map_fn, 'r')

        self.ref_fa_obj = ref_fa_obj
        self.ref_bw_obj = ref_bw_obj
        self.ref_gap_obj = load_gap()
        self.bam_obj_whole = None
        self.rb_base_chr = None
        self.rb_mappability_chr = None

    def __del__(self):
        if self.ref_fa_obj:
            self.ref_fa_obj.close()
        if self.ref_bw_obj:
            self.ref_bw_obj.close()
        if self.bam_obj_whole:
            self.bam_obj_whole.close()

    def load_bam(self, bam_fn):
        logger.info('Loading bam file...')
        bam_obj = pysam.AlignmentFile(bam_fn, mode='rb')
        self.bam_obj_whole = bam_obj

    def run(self, sample_id, chr_id, win_size=1000, min_r=0.1, stride_size=200):
        """

        :param sample_id:
        :param chr_id:
        :param win_size:
        :param min_r:
        :param stride_size:
        :return:
        """

        ref_gap_chr = self.ref_gap_obj.loc[self.ref_gap_obj['CHROM'] == 'chr' + chr_id,
                                           ['START', 'END']] if not self.ref_gap_obj.empty else None

        # reference base
        logger.info('Loading reference sequence for sample {} chr: {}...'.format(sample_id, chr_id))
        self.rb_base_chr = self.ref_fa_obj.fetch('chr' + chr_id)
        chr_len = len(self.rb_base_chr)

        # reference mappabillity
        logger.info('Loading reference mappability for sample {} chr: {}...'.format(sample_id, chr_id))
        self.rb_mappability_chr = self.ref_bw_obj.values('chr' + chr_id, 0, chr_len - 1)

        fil_pos = np.array([], dtype=np.int)
        if ref_gap_chr is not None:
            for _, i_row in ref_gap_chr.iterrows():
                # END is excluded
                fil_pos = np.concatenate((fil_pos, np.arange(i_row['START'], i_row['END'])))

        rb_base_pos = np.ones(chr_len, dtype=int)
        rb_base_pos[fil_pos] = 0

        seg_values, seg_starts, seg_lengths = find_seg(rb_base_pos)
        assert len(seg_values) == len(seg_starts) == len(seg_lengths)

        for val_idx, i_val in enumerate(seg_values):
            if i_val == 0:
                # gap region
                yield seg_starts[val_idx], seg_starts[val_idx] + seg_lengths[val_idx], seg_lengths[val_idx], 0, None
            else:
                i_seg_start = seg_starts[val_idx]
                i_seg_len = seg_lengths[val_idx]

                if i_seg_len >= win_size:
                    i_start_indices, remain_len = seq_slide(i_seg_len, win_size, stride_size)
                    for i in i_start_indices:
                        i_w_start = i+i_seg_start
                        i_w_end = i_w_start + win_size

                        yield self._get_feats_region(chr_id, i_w_start, i_w_end, win_size, min_r)

                    if remain_len > 0:
                        i_w_start = i_seg_len-win_size
                        i_w_end = i_seg_len
                        yield self._get_feats_region(chr_id, i_w_start, i_w_end, win_size, min_r)
                else:
                    yield self._get_feats_region(chr_id, i_seg_start, i_seg_start+i_seg_len, win_size, min_r)

    def _get_feats_region(self, chr_id, reg_start, reg_end, win_size, min_r):
        """

        :param chr_id:
        :param reg_start:
        :param reg_end:
        :param win_size:
        :param min_r:
        :return:
        """
        i_rb_base = self.rb_base_chr[reg_start:reg_end]
        i_ref_map = self.rb_mappability_chr[reg_start:reg_end]

        i_pileup = self.bam_obj_whole.pileup('chr' + chr_id, start=reg_start,
                                             stop=reg_end, stepper='nofilter',
                                             min_base_quality=0, truncate=True)
        ref_rel_pos, f_mat = gen_feat_region(i_pileup, i_rb_base, i_ref_map, reg_start, win_size)

        if len(ref_rel_pos) == 0:
            # no reads cover
            return reg_start, reg_end, win_size, 1, None
        elif len(ref_rel_pos) < win_size * min_r:
            # the number of reads coverage is not enough
            return reg_start, reg_end, win_size, 2, None
        else:
            # ok, requirments satisfied
            return reg_start, reg_end, win_size, 3, f_mat





