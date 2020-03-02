#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: feat_generator.py
    Description:
    
Created by Yong Bai on 2019/8/13 4:48 PM.
"""
import os
import numpy as np
import pandas as pd
import pysam
import pyBigWig
import gc
import logging

from cnv_utils.general_utils import find_seg
from cnv_utils.feat_gen_utils import load_gap, gen_feat_region

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def gen_cnv_feats(sample_id, chr_id, rb_base_chr, rb_mappability_chr, bam_obj_whole,
                  cnv_regions_chr, out_chr_fname):
    """

    :param sample_id:
    :param chr_id:
    :param rb_base_chr:
    :param rb_mappability_chr:
    :param bam_obj_whole:
    :param cnv_regions_chr:
    :param out_chr_fname:
    :return:
    """

    # ['CHROM','ID' ,'POS', 'END','SVLEN', 'ALT', 'SVTYPE','AF','SMP_GT', 'ADJTYPE','ADJAF']
    logger.info('Working on sample {} chr {}, the number of cnvs {}...'.format(
        sample_id, chr_id, len(cnv_regions_chr)))

    for index, row in cnv_regions_chr.iterrows():
        cnv_start = row['POS'] - 1
        cnv_end = row['END'] - 1  # not include
        cnv_type = row['ADJTYPE']
        cnv_af = row['ADJAF']
        cnv_len = row['SVLEN']

        its = bam_obj_whole.pileup('chr' + chr_id, start=cnv_start,
                                   stop=cnv_end, stepper='nofilter',
                                   min_base_quality=0, truncate=True)
        i_rb_base = rb_base_chr[cnv_start:cnv_end]
        i_ref_map = rb_mappability_chr[cnv_start:cnv_end]
        ref_rel_pos, f_mat = gen_feat_region(its, i_rb_base, i_ref_map, cnv_start, cnv_len)

        with open(out_chr_fname, 'a') as f:
            f.write(
                '#{},{},{},{},{},{},{}\n'.format(chr_id, cnv_start, cnv_end, cnv_len, cnv_type, cnv_af, ref_rel_pos))
            np.savetxt(f, f_mat, fmt='%-10.5f')
        del f_mat
        del ref_rel_pos
    return 'Sample {} chr {}: cnv features written to file'.format(sample_id, chr_id)


def gen_neu_feats(sample_id, chr_id, rb_base_chr, rb_mappability_chr, bam_obj_whole,
                  gap_regions_chr, cnv_regions_chr, out_chr_fname, min_reg_len=2000, n_regions=4):
    """

    :param sample_id:
    :param chr_id:
    :param rb_base_chr:
    :param rb_mappability_chr:
    :param bam_obj_whole:
    :param gap_regions_chr:
    :param cnv_regions_chr:
    :param out_chr_fname:
    :param min_reg_len:
    :param n_regions:
    :return:
    """
    # find the regions from gap and/or cnvs to be excluded
    # gaps
    fil_pos = np.array([], dtype=np.int)
    if gap_regions_chr:
        for _, i_row in gap_regions_chr.iterrows():
            # END is excluded
            fil_pos = np.concatenate((fil_pos, np.arange(i_row['START'], i_row['END'])))
    # cnv region
    # NEU features need to exclude the cnv region
    # but for prediction, cnv regions are not needed to be excluded.
    if cnv_regions_chr:
        for _, i_row in cnv_regions_chr.iterrows():
            # VCF POS is started with 1, END is also excluded
            fil_pos = np.concatenate((fil_pos, np.arange(i_row['POS'] - 1, i_row['END'] - 1)))

    rb_base_pos = np.ones(len(rb_base_chr), dtype=int)
    rb_base_pos[fil_pos] = 0

    logger.info('finding the regions to be generated feature matrix, sample {} chr {}...'.format(
        sample_id, chr_id))
    seg_values, seg_starts, seg_lengths = find_seg(rb_base_pos)

    assert len(seg_values) == len(seg_starts) == len(seg_lengths)

    fil_val_idx = np.where(seg_values == 1)[0]
    fil_len_idx = np.where(seg_lengths >= min_reg_len)[0]
    fil_idx = list(set(fil_val_idx).intersection(set(fil_len_idx)))

    t_seg_start = seg_starts[fil_idx]
    t_seg_len = seg_lengths[fil_idx]
    frt_idxs = np.argsort(t_seg_len)[:n_regions]

    logger.info('extracting features for neu region len={}, sample {} chr {}...'.format(
        t_seg_len[frt_idxs], sample_id, chr_id))

    for i_idx in frt_idxs:
        i_start = t_seg_start[i_idx]
        i_len = t_seg_len[i_idx]
        i_end = i_start + i_len
        i_rb_base = rb_base_chr[i_start:i_end]
        i_ref_map = rb_mappability_chr[i_start:i_end]
        i_pileup = bam_obj_whole.pileup('chr' + chr_id, start=i_start,
                                        stop=i_end, stepper='nofilter',
                                        min_base_quality=0, truncate=True)
        ref_rel_pos, f_mat = gen_feat_region(i_pileup, i_rb_base, i_ref_map, i_start, i_len)

        with open(out_chr_fname, 'a') as f:
            f.write('#{},{},{},{},{},{},{}\n'.format(chr_id, i_start, i_end, i_len, 'NEU', 10,
                                                     ref_rel_pos))
            np.savetxt(f, f_mat, fmt='%-10.5f')
        del f_mat
        del ref_rel_pos
    return 'Sample {} chr {}: neu features written to file'.format(sample_id, chr_id)


class FeatureGenerator:

    def __init__(self, ref_fasta_fn, ref_map_fn):
        """

        :param ref_fasta_fn: reference fasta file name
        :param ref_map_fn: reference mappability file name
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

    def __del__(self):
        if self.ref_fa_obj:
            self.ref_fa_obj.close()
        if self.ref_bw_obj:
            self.ref_bw_obj.close()

    def run(self, sample_id, chr_id_lst, bam_fn, out_fn, feat_type='neu',
            cnvs_regions_fn=None, min_seg_len=2000, n_regions=4):
        """

        :param sample_id:
        :param chr_id_lst:
        :param bam_fn:
        :param out_fn:
        :param feat_type: 'neu' or 'cnv'
        :param cnvs_regions_fn: if feat_type = 'cnv', cnvs_regions_fn must not be None
        :param min_seg_len: if feat_type = 'neu', cnvs_regions_fn, min_seg_len and n_regions should be provided.
        :param n_regions: if feat_type = 'neu', n_regions is the first n shortest neu regsions would be used to
                        extract features for NEU.
        :return:
        """

        if feat_type not in ['neu', 'cnv']:
            raise Exception('feature generator type must be neu or cnv.')

        # load cnvs regions
        cnvs_regions_whole = pd.read_csv(cnvs_regions_fn, sep='\t') if os.path.exists(cnvs_regions_fn) else None

        if (feat_type == 'cnv') and (not cnvs_regions_whole):
            raise Exception('cnvs_regions_fn is None when generate feature for cnvs.')

        if not os.path.exists(bam_fn):
            raise FileNotFoundError('Bam file does not exist. {}'.format(bam_fn))

        logger.info('Loading bam file...')
        bam_obj = pysam.AlignmentFile(bam_fn, mode='rb')

        for chr_id in chr_id_lst:
            ref_gap_chr = self.ref_gap_obj.loc[self.ref_gap_obj['CHROM'] == 'chr' + chr_id, ['START', 'END']] \
                if self.ref_gap_obj else None
            cnvs_regions_chr = cnvs_regions_whole.loc[cnvs_regions_whole['CHROM'] == chr_id, ['POS', 'END']] \
                if cnvs_regions_whole else None

            # reference base
            logger.info('Loading reference sequence for sample {} chr: {}...'.format(sample_id, chr_id))
            rb_base = self.ref_fa_obj.fetch('chr' + chr_id)
            chr_len = len(rb_base)

            # reference mappabillity
            logger.info('Loading reference mappability for sample {} chr: {}...'.format(sample_id, chr_id))
            rb_mappability = self.ref_bw_obj.values('chr' + chr_id, 0, chr_len - 1)

            logger.info('Submitting task for sample {} chr: {}...'.format(sample_id, chr_id))
            if feat_type == 'neu':
                logger.info(gen_neu_feats(sample_id, chr_id, rb_base, rb_mappability, bam_obj,
                                          ref_gap_chr, cnvs_regions_chr, out_fn, min_seg_len, n_regions))
            elif feat_type == 'cnv':
                logger.info(gen_cnv_feats(sample_id, chr_id, rb_base, rb_mappability, bam_obj,
                                          cnvs_regions_chr, out_fn))
            del cnvs_regions_chr
            del rb_base
            del rb_mappability
            gc.collect()

        if bam_obj:
            bam_obj.close()
