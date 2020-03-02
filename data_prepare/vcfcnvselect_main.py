#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: vcfcnvselect_main.py
    Description:
    
Created by Yong Bai on 2019/8/13 11:19 AM.
"""

import os
import argparse
import pandas as pd


def main(args):
    """

    :param args:
    :return:
    """
    parsed_cvf_fn = args.parsed_cvf_fn
    out_dir = args.out_dir
    cnv_types = args.cnv_types

    if not os.path.exists(parsed_cvf_fn):
        raise FileNotFoundError('Parsed vcf file does not exists. {}'.format(parsed_cvf_fn))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    cnv_types_lst = cnv_types.upper().split(',')

    print('Loading parsed vcf file...')
    data_chunks = pd.read_csv(parsed_cvf_fn, sep='\t', chunksize=2000)
    vcf_data_df_lst = []
    for chuk in data_chunks:
        vcf_data_df_lst.append(chuk)
    vcf_data_df = pd.concat(vcf_data_df_lst, ignore_index=True)

    print('Selecting target CNV types...')
    # find CNVs
    cnv_final_df = vcf_data_df.loc[vcf_data_df['SVTYPE'].isin(cnv_types_lst)]
    final_cnv_fname = os.path.join(out_dir, 'ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.cnvs.dat')
    cnv_final_df.to_csv(final_cnv_fname, sep='\t', index=False)
    print('Done, result saved at {}'.format(final_cnv_fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='select cnv types from parsed cnv files')
    parser.add_argument(
        "--parsed_cvf_fn",
        type=str,
        default="/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/cnv_data/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.parallel.NosamplesGT.dat",
        help="parsed vcf file")
    parser.add_argument(
        '--out_dir',
        type=str,
        default="/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/cnv_data",
        help="directory for saving the select CNVs data")

    parser.add_argument(
        '--cnv_types',
        type=str,
        default="CNV,DUP,DEL",
        help="CNV types to be selected, types are separated by ',' without space")

    args = parser.parse_args()
    main(args)
