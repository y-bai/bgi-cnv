#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: vcfreader_main.py
    Description:
    
Created by Yong Bai on 2019/8/13 10:28 AM.
"""
import os
import argparse
from cnv_utils import str2bool
from vcfparse import VcfReader


def main(args):
    """

    :param args:
    :return:
    """
    vcf_fn = args.vcf_fn
    is_sample_gt_dump = args.is_sampleGT_dump
    out_dir = args.out_dir

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    if is_sample_gt_dump:
        vcf_data_fname = os.path.join(out_dir, 'ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.parallel.dat')
    else:
        vcf_data_fname = os.path.join(out_dir,
                                      'ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.parallel.NosamplesGT.dat')
    if os.path.exists(vcf_data_fname):
        raise Exception('cvf parser terminated, Parsed file exists: {}'.format(vcf_data_fname))
    print('Start writing data to file...')
    vcfrd = VcfReader(vcf_fn, chunk_size=1000, dump_sample_gt=is_sample_gt_dump)
    data_df = vcfrd.run()
    data_df.to_csv(vcf_data_fname, sep='\t', index=False)
    print('Done, vcf data was saved as {}'.format(vcf_data_fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse cnv files')
    parser.add_argument(
        "--vcf_fn",
        type=str,
        default="/ifs1/pub/database/ftp.ncbi.nih.gov/1000genomes/ftp/phase3/integrated_sv_map/supporting/GRCh38_positions/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.gz",
        help="cnv file name, the reference is hg38")
    parser.add_argument(
        '--is_sampleGT_dump',
        type=str2bool,
        default=False,
        help="is needed to dump sample GT separately")
    parser.add_argument(
        '--out_dir',
        type=str,
        default="/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/cnv_data",
        help="directory for saving the parsed vcf data")

    args = parser.parse_args()
    main(args)
