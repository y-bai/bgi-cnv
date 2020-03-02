#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: mat_cnvs_data_prepare.py
    Description:
    
Created by Yong Bai on 2019/8/13 12:27 PM.
"""

import os
import pysam
import numpy as np
import pandas as pd

from collections import Counter
import pyBigWig

import argparse
import logging
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def gen_feat_chr(chr_id, chr_cnvs_df, rb_base, rb_mappability, bamfile, sample_id, out_fname):
    """

    :param chr_id:
    :param chr_cnvs_df:
    :param rb_base:
    :param rb_mappability:
    :param bamfile:
    :param sample_id:
    :param out_fname:
    :return:
    """
    # ['CHROM','ID' ,'POS', 'END','SVLEN', 'ALT', 'SVTYPE','AF','SMP_GT', 'ADJTYPE','ADJAF']
    logger.info('Working on sample {} chr {}, the number of cnvs {}...'.format(sample_id, chr_id, len(chr_cnvs_df)))

    for index, row in chr_cnvs_df.iterrows():

        cnv_start = row['POS'] - 1
        cnv_end = row['END'] - 1  # not include
        cnv_type = row['ADJTYPE']
        cnv_af = row['ADJAF']
        cnv_len = row['SVLEN']

        its = bamfile.pileup('chr' + chr_id, start=cnv_start,
                             stop=cnv_end, stepper='nofilter',
                             min_base_quality=0, truncate=True)

        f_mat = np.zeros((13, cnv_len))  # there should be 13
        # Coordinates in pysam are always 0-based (following the python convention,same to bam).
        # SAM text files use 1-based coordinates.
        ref_rel_pos = []
        for pileupcolumn in its:
            # each iteration return a PileupColumn Object which represents all the reads in the bam files that
            # map to a single base in the reference sequence. The list of reads are represented as PileupRead objects
            # in the PileupColumn.pileups property.
            ref_i_pos = pileupcolumn.reference_pos - cnv_start
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
        f_mat[8] = rb_mappability[cnv_start:cnv_end]
        # reference a,t,c,g
        for m, f_base in enumerate(rb_base[cnv_start:cnv_end]):
            if f_base.upper() == 'A':
                f_mat[9, m] = 1
            elif f_base.upper() == 'T':
                f_mat[10, m] = 1
            elif f_base.upper() == 'C':
                f_mat[11, m] = 1
            elif f_base.upper() == 'G':
                f_mat[12, m] = 1

        with open(out_fname, 'a') as f:
            f.write(
                '#{},{},{},{},{},{},{}\n'.format(chr_id, cnv_start, cnv_end, cnv_len, cnv_type, cnv_af, ref_rel_pos))
            np.savetxt(f, f_mat, fmt='%-10.5f')
        del f_mat

    del chr_cnvs_df
    gc.collect()
    return 'Sample {} chr {}: cnv features written to file'.format(sample_id, chr_id)


def main(args):

    sampleId = args.sample_id
    bam_fn = args.bam_fn
    out_dir = args.out_dir

    ref_fa_f = args.ref_fa_f
    ref_map_f = args.ref_map_f

    cnv_label_fn = args.cnv_labels_fn

    if not os.path.isdir(out_dir):
        raise Exception('Directory not exist: {}'.format(out_dir))

    if not os.path.exists(cnv_label_fn):
        raise Exception('File not exist: {}'.format(cnv_label_fn))
    cnvs_df_label = pd.read_csv(cnv_label_fn, sep='\t')

    reffile = pysam.FastaFile(ref_fa_f)
    bw = pyBigWig.open(ref_map_f, 'r')

    logger.info('Processing sample {} '.format(sampleId))
    if not os.path.exists(bam_fn):
        raise Exception('File not exist: {}'.format(bam_fn))
    bamfile = pysam.AlignmentFile(bam_fn, mode='rb')

    cnvfeat_fname = os.path.join(out_dir,
                                 'ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.' + sampleId + '.cnvs.features')

    if os.path.exists(cnvfeat_fname):
        os.remove(cnvfeat_fname)

    chr_lst_idx = [str(i) for i in range(1, 23)] + ['X']

    for i_chr in chr_lst_idx:
        logger.info('Generating featues for sample {} chr: {}...'.format(sampleId, i_chr))
        # labeled CNV for given chromosome
        logger.info('Loading CNV labels for sample {} chr: {}...'.format(sampleId, i_chr))
        chr_cnvs_df = cnvs_df_label[cnvs_df_label['CHROM'] == i_chr]
        # reference base
        logger.info('Loading reference sequence for sample {} chr: {}...'.format(sampleId, i_chr))
        rb_base = reffile.fetch('chr' + i_chr)
        chr_len = len(rb_base)
        # reference mappabillity
        logger.info('Loading reference mappability for sample {} chr: {}...'.format(sampleId, i_chr))
        rb_mappability = bw.values('chr' + i_chr, 0, chr_len)
        logger.info('Submitting task for sample {} chr: {}...'.format(sampleId, i_chr))

        logger.info(gen_feat_chr(i_chr, chr_cnvs_df, rb_base, rb_mappability, bamfile, sampleId, cnvfeat_fname))

    logger.info('output dir: %s', out_dir)

    reffile.close()
    bw.close()
    bamfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate cnv features')
    parser.add_argument('--sample_id', type=str, help="sample id")
    parser.add_argument('--bam_fn', type=str, help="bam file name")
    parser.add_argument('--out_dir', type=str,
                        default="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/1k_cnvs_lab_feat/cnvs_feat",
                        help="output directory")
    parser.add_argument('--cnv_labels_fn', type=str, help="cnv label file name")
    parser.add_argument('--ref_fa_f', type=str, help='reference fasta file', required=True)
    parser.add_argument('--ref_map_f', type=str, help='reference mappability bw file', required=True)
    args = parser.parse_args()
    main(args)

