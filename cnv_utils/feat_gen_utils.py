#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: feat_gen_utils.py
    Description:
    
Created by Yong Bai on 2019/8/14 2:15 PM.
"""
import os
import numpy as np
import pandas as pd
from collections import Counter

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_gap(gap_fn='../hg38_config/gap.txt'):
    """
    load gap position for hg38.
    gap position starts with 0 index
    :param gap_fn:
    :return:
    """
    if not os.path.exists(gap_fn):
        raise FileNotFoundError('{} not found.'.format(gap_fn))

    col_names = ['BIN', 'CHROM', 'START', 'END', 'IX', 'N', 'SIZE', 'TYPE', 'BRIDGE']
    col_dtype = [int, str, int, int, int, str, int, str, str]
    gap_header = dict(zip(col_names, col_dtype))
    gap_pd = pd.read_csv(gap_fn, sep='\t', header=None, names=col_names, dtype=gap_header)

    return gap_pd


def load_centro(centro_fn='../hg38_config/hg38_centromeres.txt'):
    """
    load centromeres position for hg38.
    gap position starts with 0 index
    :param centro_fn:
    :return:
    """
    if not os.path.exists(centro_fn):
        raise FileNotFoundError('{} not found.'.format(centro_fn))

    col_names = ['BIN', 'CHROM', 'START', 'END', 'NAME']
    col_dtype = [int, str, int, int, str]
    centro_header = dict(zip(col_names, col_dtype))
    centro_pd = pd.read_csv(centro_fn, sep='\t', header=None, names=col_names, dtype=centro_header)

    return centro_pd


def load_sup_dups(sup_dups_fn='../hg38_config/hg38_genomicSuperDups.txt'):
    """
    load centromeres position for hg38.
    gap position starts with 0 index
    :param sup_dups_fn:
    :return:
    """
    if not os.path.exists(sup_dups_fn):
        raise FileNotFoundError('{} not found.'.format(sup_dups_fn))

    col_names = ['BIN', 'CHROM', 'START', 'END', 'NAME', 'SCORE', 'STRAND',
                 'OTHERCHROM', 'OTHERSTART', 'OTHEREND', 'OTHERSIZE', 'UID',
                 'POSBASESHIT', 'TESTRESULT', 'VERDICT', 'CHITS', 'CCOV', 'ALIGNFILE',
                 'ALGINL', 'INDELN', 'INDELS', 'ALIGNB', 'MATCHB', 'MISMATCHB',
                 'TRANSITIONSB', 'TRANSVERSIONB', 'FRACMATCH', 'FRACMATCHINDEL',
                 'JCK', 'K2K']
    col_dtype = [int, str, int, int, str, float, str,
                 str, float, float, float, float,
                 float, str, str, str, str, str,
                 float, float, float, float, float, float,
                 float, float, float, float,
                 float, float]
    sup_dups_header = dict(zip(col_names, col_dtype))
    sup_dups_pd = pd.read_csv(sup_dups_fn, sep='\t', header=None, names=col_names, dtype=sup_dups_header)

    out_cols = ['BIN', 'CHROM', 'START', 'END', 'NAME']
    sup_dups_pd = sup_dups_pd[out_cols]
    return sup_dups_pd


def gen_feat_region(bam_pileup, rb_base, rb_mappability, reg_start, reg_len):
    """
    generate feature matrix for a given region.
    :param bam_pileup: bam_pileup object for the given region
    :param rb_base: reference sequence for the given region
    :param rb_mappability: reference mappability for the given region
    :param reg_start: region start position (absolute position from reference)
    :param reg_len: region length
    :return:
        ref_rel_pos: relative position to the reference position, index start at 0
        f_mat: feature matrix. shape = (13, reg_len)
    """
    f_mat = np.zeros((13, reg_len))
    # Coordinates in pysam are always 0-based (following the python convention,same to bam).
    # SAM text files use 1-based coordinates.
    ref_rel_pos = []
    for pileupcolumn in bam_pileup:
        # each iteration return a PileupColumn Object which represents all the reads in the bam files that
        # map to a single base in the reference sequence. The list of reads are represented as PileupRead objects
        # in the PileupColumn.pileups property.
        ref_i_pos = pileupcolumn.reference_pos - reg_start
        # save the relative position
        ref_rel_pos.append(ref_i_pos)

        # base coverage
        f_mat[0, ref_i_pos] = pileupcolumn.get_num_aligned()
        # base quality
        f_mat[1, ref_i_pos] = np.mean(pileupcolumn.get_query_qualities())
        # base map quality
        f_mat[2, ref_i_pos] = np.mean(pileupcolumn.get_mapping_qualities())

        i_seq = [x.upper() for x in
                 pileupcolumn.get_query_sequences()]  # (mark_matches=True, mark_ends=True,add_indels=True)]
        i_read_bases_cnt = Counter(i_seq)
        f_mat[3, ref_i_pos] = i_read_bases_cnt.get('G', 0) + i_read_bases_cnt.get('C', 0)
        f_mat[4, ref_i_pos] = i_read_bases_cnt.get('A', 0)
        f_mat[5, ref_i_pos] = i_read_bases_cnt.get('T', 0)
        f_mat[6, ref_i_pos] = i_read_bases_cnt.get('C', 0)
        f_mat[7, ref_i_pos] = i_read_bases_cnt.get('G', 0)

    # reference mappability
    f_mat[8] = rb_mappability
    # reference a,t,c,g
    for m, f_base in enumerate(rb_base):
        if f_base.upper() == 'A':
            f_mat[9, m] = 1
        elif f_base.upper() == 'T':
            f_mat[10, m] = 1
        elif f_base.upper() == 'C':
            f_mat[11, m] = 1
        elif f_base.upper() == 'G':
            f_mat[12, m] = 1

    return ref_rel_pos, f_mat






