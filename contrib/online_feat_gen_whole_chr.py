#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: online_feat_gen_whole_chr.py
    Description: generate feature maps for online cnv call (CPU only)
    
Created by Yong Bai on 2019/9/23 11:01 AM.
"""

import os
import numpy as np

import pysam
import pyBigWig
import logging
import concurrent.futures
import multiprocessing
from collections import Counter

from cnv_utils import gen_feat_whole_chr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def gen_feat_single(pileup_column):
    """
    concurrent generating feature for a single pileup column by pysam.pileup() function
    :param pileup_column:
    :return:
    """
    # Coordinates in pysam are always 0-based (following the python convention,same to bam).
    # SAM text files use 1-based coordinates.

    # each iteration return a PileupColumn Object which represents all the reads in the bam files that
    # map to a single base in the reference sequence. The list of reads are represented as PileupRead objects
    # in the PileupColumn.pileups property.
    ref_i_pos = pileup_column.reference_pos

    # if ref_i_pos % 1000000 == 0:
    #     logger.info('feature generation at position {}'.format(ref_i_pos))

    # base coverage
    base_cov = pileup_column.get_num_aligned()
    # base quality
    base_qual = np.mean(pileup_column.get_query_qualities())
    # base map quality
    base_map_qual = np.mean(pileup_column.get_mapping_qualities())

    i_seq = [x.upper() for x in
             pileup_column.get_query_sequences()]  # (mark_matches=True, mark_ends=True,add_indels=True)]
    i_read_bases_cnt = Counter(i_seq)
    base_gc_cnt = i_read_bases_cnt.get('G', 0) + i_read_bases_cnt.get('C', 0)
    base_a_cnt = i_read_bases_cnt.get('A', 0)
    base_t_cnt = i_read_bases_cnt.get('T', 0)
    base_c_cnt = i_read_bases_cnt.get('C', 0)
    base_g_cnt = i_read_bases_cnt.get('G', 0)

    return ref_i_pos, base_cov, base_qual, base_map_qual, base_gc_cnt, base_a_cnt, base_t_cnt, base_c_cnt, base_g_cnt


class OnlineFeatureGenWholeChr:

    def __init__(self, sample_id, bam_fn, ref_fasta_fn, ref_map_fn, fmt='bam'):

        self.sample_id = sample_id
        if not os.path.exists(ref_fasta_fn):
            raise FileNotFoundError('Reference fasta file does not exist. {}'.format(ref_fasta_fn))
        if not os.path.exists(ref_map_fn):
            raise FileNotFoundError('Reference mappability bw file does not exist. {}'.format(ref_map_fn))
        if not os.path.exists(bam_fn):
            raise FileNotFoundError('bam file does not exist. {}'.format(ref_map_fn))

        logger.info('loading Reference fasta file...')
        self.ref_fa_obj = pysam.FastaFile(ref_fasta_fn)
        logger.info('loading Reference mappability bw file...')
        self.ref_bw_obj = pyBigWig.open(ref_map_fn, 'r')
        logger.info('Loading bam file...')
        if fmt == 'bam':
            self.bam_obj_whole = pysam.AlignmentFile(bam_fn, mode='rb')

    def __del__(self):

        if self.ref_fa_obj is not None:
            self.ref_fa_obj.close()
        if self.ref_bw_obj is not None:
            self.ref_bw_obj.close()
        if self.bam_obj_whole is not None:
            self.bam_obj_whole.close()

    def cal_feat_whole(self, chr_id, out_feat_whole_chr_fn, n_features=13, n_proc=32):

        # reference base
        logger.info('Loading reference sequence for sample {} chr: {}...'.format(self.sample_id, chr_id))
        rb_base_chr = self.ref_fa_obj.fetch('chr' + chr_id)
        chr_len = len(rb_base_chr)

        # reference mappabillity
        logger.info('Loading reference mappability for sample {} chr: {}...'.format(self.sample_id, chr_id))
        rb_mappability_chr = self.ref_bw_obj.values('chr' + chr_id, 0, chr_len)

        # call pysam pileup() to read the whole chr
        logger.info('loading whole bam pileup for sample {} chr: {}...'.format(self.sample_id, chr_id))
        assert self.bam_obj_whole is not None
        whole_chr_pileup = self.bam_obj_whole.pileup('chr' + chr_id, stepper='nofilter', min_base_quality=0)

        # get the feature map for the whole chr
        # take almost 25 minutes for  chr1
        logger.info('calculating feature maps for whole chromosome, sample {}, chr {}...'.format(
            self.sample_id, chr_id))
        logger.info('>>>>>>>>>>this processing will take a few minutes (almost 180 minutes for chr 1)...')
        # chr_feat_mat_whole = gen_feat_whole_chr(whole_chr_pileup,
        #                                         rb_base_chr,
        #                                         chr_len,
        #                                         rb_mappability_chr)

        chr_feat_mat_whole = np.zeros((chr_len, n_features))
        logger.info('launch ing multiprocessing, proc={}...'.format(n_proc))
        p = multiprocessing.Pool(n_proc)  # multiprocessing.cpu_count()
        res = p.map(gen_feat_single, whole_chr_pileup)
        for i_re in res:
            i_re_pos, *i_re_info = i_re
            if i_re_pos % 1000000 == 0:
                logger.info('processing at {}'.format(i_re_pos))
        p.close()
        p.join()
            # # base coverage
            # chr_feat_mat_whole[i_ref_pos, 0] = i_re_tup[0]
            # # base quality
            # chr_feat_mat_whole[i_ref_pos, 1] = i_re_tup[1]
            # # base map quality
            # chr_feat_mat_whole[i_ref_pos, 2] = i_re_tup[2]
            # # base_gc_cnt
            # chr_feat_mat_whole[i_ref_pos, 3] = i_re_tup[3]
            # # base_a_cnt
            # chr_feat_mat_whole[i_ref_pos, 4] = i_re_tup[4]
            # # base_t_cnt
            # chr_feat_mat_whole[i_ref_pos, 5] = i_re_tup[5]
            # # base_c_cnt
            # chr_feat_mat_whole[i_ref_pos, 6] = i_re_tup[6]
            # # base_g_cnt
            # chr_feat_mat_whole[i_ref_pos, 7] = i_re_tup[7]
            #
            # # reference mappability
            # chr_feat_mat_whole[i_ref_pos, 8] = rb_mappability_chr[i_ref_pos]
            #
            # chr_feat_mat_whole[i_ref_pos, 9] = 1 if rb_base_chr[i_ref_pos] == 'A' else 0
            # chr_feat_mat_whole[i_ref_pos, 10] = 1 if rb_base_chr[i_ref_pos] == 'T' else 0
            # chr_feat_mat_whole[i_ref_pos, 11] = 1 if rb_base_chr[i_ref_pos] == 'C' else 0
            # chr_feat_mat_whole[i_ref_pos, 12] = 1 if rb_base_chr[i_ref_pos] == 'G' else 0

        # logger.info('saving the results...')
        # with open(out_feat_whole_chr_fn, 'w') as f:
        #     np.savetxt(f, chr_feat_mat_whole, fmt='%-10.5f')
