#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: online_feat_gen_whole_chr_main.py
    Description:
    shared variable using multiprocessing
    
Created by Yong Bai on 2019/9/23 11:37 AM.
"""

import os
import numpy as np

import pysam
import pyBigWig
from collections import Counter
import multiprocessing
import time
import h5py
import ctypes

import argparse
import logging
import sys
sys.path.append("..")
from cnv_utils.general_utils import seq_slide

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# global variable
# regularly, each sub process will have one copy
rb_base_chr = None
rb_mappability_chr = None
bam_fn = ''
chr_id = ''


def mp_init(l):
    global lock
    lock = l


def multi_run_wrapper(single_args):
    return gen_feat_whole_chr(*single_args)


def gen_feat_whole_chr(chunk_start, chunck_len):
    lock.acquire()
    f_mat = np.zeros((chunck_len, 13), dtype=np.float32)
    base_cov_pos = np.zeros(chunck_len, dtype=np.int)

    # Coordinates in pysam are always 0-based (following the python convention,same to bam).
    # SAM text files use 1-based coordinates.
    # multiple processing requires opening large file here
    logger.info('loading bam file for chunk start {}...'.format(chunk_start))
    bam_obj = pysam.AlignmentFile(bam_fn, mode='rb')
    i_pileup = bam_obj.pileup('chr' + chr_id, start=chunk_start,
                              stop=chunk_start + chunck_len, stepper='nofilter',
                              min_base_quality=0, truncate=True)
    logger.info('calculating features for chunk start={}, len={}...'.format(chunk_start, chunck_len))
    # Coordinates in pysam are always 0-based (following the python convention,same to bam).
    # SAM text files use 1-based coordinates.
    for pileupcolumn in i_pileup:
        # each iteration return a PileupColumn Object which represents all the reads in the bam files that
        # map to a single base in the reference sequence. The list of reads are represented as PileupRead objects
        # in the PileupColumn.pileups property.
        ref_i_pos = pileupcolumn.reference_pos - chunk_start
        base_cov_pos[ref_i_pos] = 1

        # base coverage
        f_mat[ref_i_pos, 0] = pileupcolumn.get_num_aligned()
        # base quality
        f_mat[ref_i_pos, 1] = np.mean(pileupcolumn.get_query_qualities())
        # base map quality
        f_mat[ref_i_pos, 2] = np.mean(pileupcolumn.get_mapping_qualities())

        i_seq = [x.upper() for x in
                 pileupcolumn.get_query_sequences()]  # (mark_matches=True, mark_ends=True,add_indels=True)]
        i_read_bases_cnt = Counter(i_seq)
        f_mat[ref_i_pos, 3] = i_read_bases_cnt.get('G', 0) + i_read_bases_cnt.get('C', 0)
        f_mat[ref_i_pos, 4] = i_read_bases_cnt.get('A', 0)
        f_mat[ref_i_pos, 5] = i_read_bases_cnt.get('T', 0)
        f_mat[ref_i_pos, 6] = i_read_bases_cnt.get('C', 0)
        f_mat[ref_i_pos, 7] = i_read_bases_cnt.get('G', 0)

    del i_pileup
    del bam_obj

    # reference mappability
    f_mat[:, 8] = rb_mappability_chr[chunk_start:chunk_start + chunck_len]
    # f_mat[:, 8] = np.frombuffer(rb_mappability_chr.get_obj())[chunk_start:chunk_start + chunck_len]
    # or f_mat[:, 8] = np.ctypeslib.as_array(rb_mappability_chr.get_obj())
    #
    # reference a,t,c,g
    rb_base_chr_str = str(rb_base_chr.value.decode('utf-8'))
    for m, f_base in enumerate(rb_base_chr_str[chunk_start:chunk_start + chunck_len]):
        if f_base.upper() == 'A':
            f_mat[m, 9] = 1
        elif f_base.upper() == 'T':
            f_mat[m, 10] = 1
        elif f_base.upper() == 'C':
            f_mat[m, 11] = 1
        elif f_base.upper() == 'G':
            f_mat[m, 12] = 1

    lock.release()
    return chunk_start, chunck_len, base_cov_pos, f_mat


def main(args):
    sample_id = args.sample_id

    global chr_id
    chr_id = args.chr_id

    in_bam_fn = args.in_bam_fn
    ref_fa_f = args.ref_fa_f
    ref_map_f = args.ref_map_f

    n_features = args.n_features
    n_proc = args.n_proc

    fmt = args.fmt

    online_out_root_dir = args.online_out_root_dir

    online_out_sample_subdir = os.path.join(online_out_root_dir, sample_id)
    if not os.path.isdir(online_out_sample_subdir):
        os.mkdir(online_out_sample_subdir)

    online_out_sample_data_dir = os.path.join(online_out_sample_subdir, 'data')
    if not os.path.isdir(online_out_sample_data_dir):
        os.mkdir(online_out_sample_data_dir)

    online_out_sample_data_fn = os.path.join(online_out_sample_data_dir,
                                             '{}_chr{}_features_whole.h5'.format(sample_id, chr_id))

    if not os.path.exists(ref_fa_f):
        raise FileNotFoundError('Reference fasta file does not exist. {}'.format(ref_fa_f))
    if not os.path.exists(ref_map_f):
        raise FileNotFoundError('Reference mappability bw file does not exist. {}'.format(ref_map_f))
    if not os.path.exists(in_bam_fn):
        raise FileNotFoundError('bam file does not exist. {}'.format(in_bam_fn))

    logger.info('loading Reference fasta file...')
    ref_fa_obj = pysam.FastaFile(ref_fa_f)
    logger.info('loading Reference mappability bw file...')
    ref_bw_obj = pyBigWig.open(ref_map_f, 'r')
    logger.info('Loading bam file...')

    global bam_fn
    bam_fn = in_bam_fn

    # reference base
    logger.info('Loading reference sequence for sample {} chr: {}...'.format(sample_id, chr_id))
    # rb_base_chr_tmp = ref_fa_obj.fetch('chr' + chr_id)
    global rb_base_chr
    rb_base_chr_tmp = ref_fa_obj.fetch('chr' + chr_id)
    chr_len = len(rb_base_chr_tmp)
    logger.info('sample {}, chr {}, length {}'.format(sample_id, chr_id, chr_len))
    rb_base_chr_tmp = bytes(rb_base_chr_tmp.encode('utf-8'))
    # shared variable/object used by multiprocess
    rb_base_chr = multiprocessing.Value(ctypes.c_char_p, rb_base_chr_tmp)
    # alternative: define a mere variable/object that can be used by multiprocess
    # using multiprocess. Manager()

    # reference mappabillity
    logger.info('Loading reference mappability for sample {} chr: {}...'.format(sample_id, chr_id))
    global rb_mappability_chr
    rb_mappability_chr = multiprocessing.Array('d',
                                               ref_bw_obj.values('chr' + chr_id, 0, chr_len))

    # call pysam pileup() to read the whole chr
    logger.info('calculating feature maps for whole chromosome, sample {}, chr {}...'.format(
        sample_id, chr_id))

    chunk_len = int(1e5)
    start_p, last_start_p, last_len = seq_slide(chr_len, chunk_len, chunk_len)

    chunk_start_lens = list(zip(start_p, np.repeat(chunk_len, len(start_p))))
    if last_len > 0:
        chunk_start_lens.append((last_start_p, last_len))

    # n_proc = multiprocessing.cpu_count()  # e16-1: multiprocessing.cpu_count() = 48
    logger.info('>>>>>>>>>>this processing will take a few minutes on {} CPUs...'.format(n_proc))
    locker = multiprocessing.Lock()
    # with ThreadPool(n_proc) as p, h5py.File(online_out_sample_data_fn, 'w') as h5_out:
    with multiprocessing.Pool(n_proc, initializer=mp_init, initargs=(locker,)) as p:

        results = p.imap(multi_run_wrapper, chunk_start_lens)  #
        time.sleep(10)

        for i_chk_start, i_chk_len, i_base_pos, i_f_mat in results:
            if i_chk_start == 0:
                with h5py.File(online_out_sample_data_fn, 'w') as h5_out:
                    h5_out.create_dataset('feature', data=i_f_mat, maxshape=(None, n_features),
                                          compression="gzip", compression_opts=4)
                    h5_out.create_dataset('base_cov_pos_idx', data=i_base_pos, maxshape=(None,),
                                          compression="gzip", compression_opts=4)
                    h5_out.create_dataset('chr_len', data=chr_len,
                                          compression="gzip", compression_opts=4)
            else:
                with h5py.File(online_out_sample_data_fn, 'a') as h5_out:
                    h5_out['feature'].resize((i_chk_start + i_chk_len), axis=0)
                    h5_out['feature'][-i_chk_len:] = i_f_mat

                    h5_out['base_cov_pos_idx'].resize((i_chk_start + i_chk_len), axis=0)
                    h5_out['base_cov_pos_idx'][-i_chk_len:] = i_base_pos

            logger.info('finished at {}'.format(i_chk_start))
    logger.info('Done, whole features for {} chr {} saving at {}'.format(
        sample_id, chr_id, online_out_sample_data_fn))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate whole feature map for given sample')
    parser.add_argument(
        "-b",
        "--in_bam_fn",
        type=str,
        help="input bam file",
        required=True)

    parser.add_argument(
        "-s",
        "--sample_id",
        type=str,
        help="input sample id",
        required=True)

    parser.add_argument(
        "-c",
        "--chr_id",
        type=str,
        default='A',
        help="chromosome id,  provide like '1'")

    parser.add_argument(
        "-o",
        "--online_out_root_dir",
        type=str,
        default='/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online',
        help="online output directory")

    parser.add_argument(
        "-a",
        "--fmt",
        type=str,
        default='bam',
        help="bam file format")

    parser.add_argument(
        "-u",
        "--n_features",
        type=int,
        default=13,
        help="the number of features")

    parser.add_argument(
        "-t",
        "--n_proc",
        type=int,
        default=16,
        help="the number of processor")

    parser.add_argument("-f", '--ref_fa_f', type=str, help='reference fasta file')
    parser.add_argument("-m", '--ref_map_f', type=str, help='reference mappability bw file')
    args = parser.parse_args()
    logger.info(args)
    main(args)
