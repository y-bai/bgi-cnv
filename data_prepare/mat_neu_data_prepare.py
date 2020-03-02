#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: mat_neu_data_prepare.py
    Description:
    
Created by Yong Bai on 2019/7/30 11:28 AM.
"""
import sys
sys.path.append("..")

import os

import numpy as np
import pandas as pd
import pysam
import pyBigWig
from cnv_utils import load_gap

from collections import Counter
import gc

import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_seg(x):
    """
    ref: https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    :param x:
    :return:
    """

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find seg starts
        loc_seg_start = np.empty(n, dtype=bool)
        loc_seg_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_seg_start[1:])
        seg_starts = np.nonzero(loc_seg_start)[0]

        # find seg values
        #
        seg_values = x[loc_seg_start]

        # find seg lengths
        seg_lengths = np.diff(np.append(seg_starts, n))

        return seg_values, seg_starts, seg_lengths


def get_neu_feat_region(bam_pileup, rb_base, rb_mappability, reg_start, reg_len):
    """

    :param bam_pileup:
    :param rb_base:
    :param rb_mappability:
    :param reg_start:
    :param reg_len:
    :return:
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


def gen_neu_feat(i_chr, chr_cnvs_df, rb_base, rb_mappability, bamfile, gap_pd, min_reg_len, sampleId,
                 neufeat_fname):
    """

    :param i_chr:
    :param chr_cnvs_df:
    :param rb_base:
    :param rb_mappability:
    :param bamfile:
    :param gap_pd:
    :param min_reg_len:
    :param sampleId:
    :param neufeat_fname:
    :return:
    """
    # gaps
    filtered_pos = np.array([], dtype=np.int)
    for _, i_row in gap_pd.iterrows():
        # END is excluded
        filtered_pos = np.concatenate((filtered_pos, np.arange(i_row['START'], i_row['END'])))
    # cnv region
    for _, i_row in chr_cnvs_df.iterrows():
        # VCF POS is started with 1, END is also excluded
        filtered_pos = np.concatenate((filtered_pos, np.arange(i_row['POS'] - 1, i_row['END'] - 1)))

    rb_base_pos = np.ones(len(rb_base), dtype=int)
    rb_base_pos[filtered_pos] = 0

    logger.info('finding the neu regions, sample {} chr {}...'.format(sampleId, i_chr))
    seg_values, seg_starts, seg_lengths = find_seg(rb_base_pos)
    assert len(seg_values) == len(seg_starts) == len(seg_lengths)

    fil_val_idx = np.where(seg_values == 1)[0]
    fil_len_idx = np.where(seg_lengths >= min_reg_len)[0]
    fil_idx = list(set(fil_val_idx).intersection(set(fil_len_idx)))

    t_seg_start = seg_starts[fil_idx]
    t_seg_len = seg_lengths[fil_idx]
    frt_idxs = np.argsort(t_seg_len)[:20]  # select first 20 neu

    logger.info('extracting features for neu region len={}, sample {} chr {}...'.format(
        t_seg_len[frt_idxs], sampleId, i_chr))

    for i_idx in frt_idxs:
        i_start = t_seg_start[i_idx]
        i_len = t_seg_len[i_idx]
        i_end = i_start + i_len
        i_rb_base = rb_base[i_start:i_end]
        i_ref_map = rb_mappability[i_start:i_end]
        i_pileup = bamfile.pileup('chr' + i_chr, start=i_start,
                              stop=i_end, stepper='nofilter',
                              min_base_quality=0, truncate=True)
        ref_rel_pos, f_mat = get_neu_feat_region(i_pileup, i_rb_base, i_ref_map, i_start, i_len)

        with open(neufeat_fname, 'a') as f:
            f.write('#{},{},{},{},{},{},{}\n'.format(i_chr, i_start, i_end, i_len, 'NEU', -1,
                                                 ref_rel_pos))
            np.savetxt(f, f_mat, fmt='%-10.5f')
        del f_mat
        del ref_rel_pos

    return 'Sample {} chr {}: neu features written to file'.format(sampleId, i_chr)


def main(args):
    """

    :param args:
    :return:
    """
    sampleId = args.sample_id
    bam_fn = args.bam_fn
    out_dir = args.out_dir
    ref_fa_f = args.ref_fa_f
    ref_map_f = args.ref_map_f
    cnv_label_fn = args.cnv_labels_fn
    min_reg_len = args.min_reg_len

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # bam file
    if not os.path.exists(bam_fn):
        raise FileNotFoundError('File not exist: {}'.format(bam_fn))
    # cnv regions
    if not os.path.exists(cnv_label_fn):
        raise FileNotFoundError('File not exist: {}'.format(cnv_label_fn))
    cnvs_df_label = pd.read_csv(cnv_label_fn, sep='\t')

    # reference fasts
    if not os.path.exists(ref_fa_f):
        raise FileNotFoundError('File not exist: {}'.format(ref_fa_f))

    # reference ref_map_f
    if not os.path.exists(ref_map_f):
        raise FileNotFoundError('File not exist: {}'.format(ref_map_f))

    reffile = pysam.FastaFile(ref_fa_f)
    bw = pyBigWig.open(ref_map_f, 'r')
    bamfile = pysam.AlignmentFile(bam_fn, mode='rb')

    # load reference genome gaps
    gap_pd = load_gap()
    logger.info('Processing sample {} '.format(sampleId))
    neufeat_fname = os.path.join(out_dir,
                                 'ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.{}.{}.neus.features'.format(
                                     sampleId, min_reg_len))
    if os.path.exists(neufeat_fname):
        os.remove(neufeat_fname)

    chr_lst_idx = [str(i) for i in range(1, 23)] + ['X']

    for i_chr in chr_lst_idx:
        logger.info('Generating neu features for sample {} chr: {}...'.format(sampleId, i_chr))
        # labeled CNV for given chromosome
        logger.info('Loading CNV labels for sample {} chr: {}...'.format(sampleId, i_chr))
        chr_cnvs_df = cnvs_df_label.loc[cnvs_df_label['CHROM'] == i_chr, ['POS', 'END']]
        # reference base
        logger.info('Loading reference sequence for sample {} chr: {}...'.format(sampleId, i_chr))
        rb_base = reffile.fetch('chr' + i_chr)
        chr_len = len(rb_base)
        # reference mappabillity
        logger.info('Loading reference mappability for sample {} chr: {}...'.format(sampleId, i_chr))
        rb_mappability = bw.values('chr' + i_chr, 0, chr_len)
        logger.info('Submitting task for sample {} chr: {}...'.format(sampleId, i_chr))

        # gaps for i_chr
        i_gaps = gap_pd.loc[gap_pd['CHROM'] == 'chr' + i_chr, ['START', 'END']]

        logger.info(gen_neu_feat(i_chr,
                                 chr_cnvs_df,
                                 rb_base,
                                 rb_mappability,
                                 bamfile,
                                 i_gaps,
                                 min_reg_len,
                                 sampleId,
                                 neufeat_fname))

        del chr_cnvs_df
        del rb_base
        del rb_mappability
        gc.collect()

    logger.info('output: %s', neufeat_fname)

    reffile.close()
    bw.close()
    bamfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate cnv features')
    parser.add_argument("-s", '--sample_id', type=str, help="sample id")
    parser.add_argument("-b", '--bam_fn', type=str, help="bam file name")
    parser.add_argument("-c", '--cnv_labels_fn', type=str, help="cnv label file name")
    parser.add_argument("-o", '--out_dir', type=str, help="output directory")
    parser.add_argument("-f", '--ref_fa_f', type=str, help='reference fasta file')
    parser.add_argument("-l", '--min_reg_len', type=int, default=2000, help='min reg length to be extracted')
    parser.add_argument("-m", '--ref_map_f', type=str, help='reference mappability bw file')
    args = parser.parse_args()
    main(args)
