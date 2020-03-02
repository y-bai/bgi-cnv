# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: labeled_cnvs_preprocess.py
    Description:

Created by Yong Bai on 2019/7/26 11:18 AM.
"""

import os
import numpy as np
import pandas as pd
import argparse
import logging
import gc

import concurrent.futures

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_data(sample_id, sample_cnv_df, out_dir):
    """

    :param sample_id:
    :param sample_cnv_df:
    :param out_dir:
    :return:
    """
    logger.info(
        'Processing sample: %s, the number of cnvs: %d',
        sample_id,
        len(sample_cnv_df))
    re_str = '(' + sample_id + r',[0-9]\|?[0-9]?)'

    sample_cnv_df['SMP_GT'] = sample_cnv_df['SUPP_HETS'].str.extract(
        re_str).combine_first(sample_cnv_df['SUPP_HOM_ALTS'].str.extract(re_str))

    sample_cnv_df['ADJTYPE'] = ''
    sample_cnv_df['ADJAF'] = 0

    for i, i_row in sample_cnv_df.iterrows():
        alts = i_row['ALT'].split(',')
        afs = i_row['AF'].split(',')

        if pd.isna(i_row['SMP_GT']):
            raise Exception(
                'Sample {}: no GT given: {}'.format(
                    sample_id, i_row))

        gts = i_row['SMP_GT'].split(',')[1].split('|')
        adjalts = [alts[int(gt) - 1] for gt in gts if int(gt) > 0]
        adjafs = [float(afs[int(gt) - 1]) for gt in gts if int(gt) > 0]
        if len(set(adjalts)) == 1:
            if adjalts[0] == '<CN0>':
                sample_cnv_df.loc[i, 'ADJTYPE'] = 'DEL'
            elif '<CN' not in adjalts[0]:
                sample_cnv_df.loc[i, 'ADJTYPE'] = i_row['SVTYPE']
            else:
                sample_cnv_df.loc[i, 'ADJTYPE'] = 'DUP'
            sample_cnv_df.loc[i, 'ADJAF'] = adjafs[0]
        else:
            t_type = adjalts[np.argmax(adjafs)]
            if t_type == '<CN0>':
                sample_cnv_df.loc[i, 'ADJTYPE'] = 'DEL'
            elif '<CN' not in t_type:
                sample_cnv_df.loc[i, 'ADJTYPE'] = i_row['SVTYPE']
            else:
                sample_cnv_df.loc[i, 'ADJTYPE'] = 'DUP'
            sample_cnv_df.loc[i, 'ADJAF'] = np.max(adjafs)

    sample_cnv_df = sample_cnv_df[['CHROM',
                                   'ID',
                                   'POS',
                                   'END',
                                   'SVLEN',
                                   'ALT',
                                   'SVTYPE',
                                   'AF',
                                   'SMP_GT',
                                   'ADJTYPE',
                                   'ADJAF']]

    fcnv_fname = os.path.join(
        out_dir,
        'ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.' +
        sample_id +
        '.cnvs.labels')
    sample_cnv_df.to_csv(fcnv_fname, index=False, sep='\t')

    del sample_cnv_df
    gc.collect()
    return '{} cnv labels written to file'.format(sample_id)


def main(args):
    """

    :param args:
    :return:
    """
    in_file = args.parsed_cnv_infile
    out_dir = args.out_dir
    sample_id = args.sample_id
    sample_map_list = args.f_samseqid_mapping
    n_proc = args.n_p

    if not os.path.isdir(out_dir):
        raise Exception('Directory not exist: {}'.format(out_dir))

    if not os.path.exists(in_file):
        raise Exception('File not exist: {}'.format(in_file))

    if os.path.exists(sample_map_list):
        sample_id_map = np.loadtxt(
            sample_map_list, delimiter='\t', usecols=(
                0, 1), dtype='str')
        sample_id_arr = sample_id_map[:, 0]
    else:
        if sample_id:
            sample_id_arr = np.array([sample_id])
        else:
            raise Exception(
                'sample id is not given, or file {} does not exist!'.format(sample_map_list))

    cnvs_annos = pd.read_csv(
        in_file,
        sep='\t',
        usecols=[
            'CHROM',
            'POS',
            'ID',
            'ALT',
            'SVTYPE',
            'SVLEN',
            'END',
            'AF',
            'SUPP_N_HET',
            'SUPP_N_HOM_ALT',
            'SUPP_HETS',
            'SUPP_HOM_ALTS'])
    cnvs_annos['CHROM'] = cnvs_annos['CHROM'].astype(str)
    cnvs_annos['END'] = cnvs_annos['END'].astype(int)
    cnvs_annos['SVLEN'] = cnvs_annos['SVLEN'].astype(int)

    logger.info('Submitting tasks...')
#     mp_manager = mp.Manager()
#     lock = mp_manager.Lock()
    k = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_proc) as pool:
        res = [pool.submit(process_data, sample_id,
                           cnvs_annos[cnvs_annos['SUPP_HETS'].str.contains(sample_id)
                                      | cnvs_annos['SUPP_HOM_ALTS'].str.contains(sample_id)],
                           out_dir) for sample_id in sample_id_arr]
        for re in concurrent.futures.as_completed(res):
            logger.info('%s', re.result())
            k += 1
    logger.info('total: %d', k)
    logger.info('output dir: %s', out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate cnv labels')
    parser.add_argument(
        "--sample_id",
        type=str,
        default="NA12878",
        help="sample id")
    parser.add_argument(
        '--f_samseqid_mapping',
        type=str,
        default="",
        help="sample id and sequencing id mapping file. If given, then sample_id will be ignored")
    parser.add_argument(
        "--parsed_cnv_infile",
        type=str,
        default="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.cnvs.dat",
        help='parsed cnv file')
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/1k_cnvs_lab_feat",
        help="output directory")
    parser.add_argument("--n_p", type=int, default=1, help="Number of process")
    args = parser.parse_args()

    main(args)
