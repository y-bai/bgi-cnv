#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: online_caller.py
    Description: prediction end-to-end
    
Created by Yong Bai on 2019/9/20 1:14 PM.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import concurrent.futures
from collections import Counter

import gc
import pysam
import pyBigWig
import logging

sys.path.append("..")
from model import cnv_net
from cnv_utils.feat_gen_utils import load_gap, gen_feat_whole_chr
from cnv_utils.general_utils import find_seg, seq_slide

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

    if ref_i_pos % 10000 == 0:
        logger.info('feature generation at position {}'.format(ref_i_pos))

    return ref_i_pos, base_cov, base_qual, base_map_qual, base_gc_cnt, base_a_cnt, base_t_cnt, base_c_cnt, base_g_cnt


class OnlineCNVCall:

    def __init__(self, model_weight_fn, n_features, n_classes):

        self.n_features = n_features
        self.n_classes = n_classes
        self.ref_fasta_fn = None
        self.ref_map_fn = None

        self.ref_fa_obj = None
        self.ref_bw_obj = None
        self.ref_gap_obj = None
        self.bam_obj_whole = None

        self.rb_base_chr = None
        self.rb_mappability_chr = None

        # shape=(13 x len(self.rb_base_chr))
        self.chr_feat_mat_whole = None

        # indicator = 0,1,2
        self.chr_segs_unpredictable = []
        # indicator = 3
        self.chr_segs_predictable = []
        # feat_map when indicator=3
        self.chr_segs_predictable_feats = []

        self.sample_id = None
        self.chr_id = None

        self.model, self.batch_size = self.__load_model(model_weight_fn)

    def __del__(self):

        if self.ref_fa_obj is not None:
            self.ref_fa_obj.close()
        if self.ref_bw_obj is not None:
            self.ref_bw_obj.close()
        if self.bam_obj_whole is not None:
            self.bam_obj_whole.close()

        if self.rb_base_chr is not None:
            del self.rb_base_chr
        if self.rb_mappability_chr is not None:
            del self.rb_mappability_chr

        if self.chr_feat_mat_whole is not None:
            del self.chr_feat_mat_whole

        if len(self.chr_segs_unpredictable) > 0:
            del self.chr_segs_unpredictable
        if len(self.chr_segs_predictable) > 0:
            del self.chr_segs_predictable
        if len(self.chr_segs_predictable_feats) > 0:
            del self.chr_segs_predictable_feats

    def load_deps(self, bam_fn, ref_fasta_fn, ref_map_fn, fmt='bam'):
        """
        load depended files
        :param bam_fn: bam file, also could be .cram files
        :param ref_fasta_fn: reference fasta file
        :param ref_map_fn: reference mappability file with specified k-mer
        :param fmt: bam format: .bam file or .cram file
        """

        if not os.path.exists(ref_fasta_fn):
            raise FileNotFoundError('Reference fasta file does not exist. {}'.format(ref_fasta_fn))
        if not os.path.exists(ref_map_fn):
            raise FileNotFoundError('Reference mappability bw file does not exist. {}'.format(ref_map_fn))

        logger.info('loading Reference fasta file...')
        self.ref_fa_obj = pysam.FastaFile(ref_fasta_fn)
        logger.info('loading Reference mappability bw file...')
        self.ref_bw_obj = pyBigWig.open(ref_map_fn, 'r')
        logger.info('loading Reference gap...')
        self.ref_gap_obj = load_gap()

        logger.info('Loading bam file...')
        if fmt == 'bam':
            self.bam_obj_whole = pysam.AlignmentFile(bam_fn, mode='rb')

    def cal_feat_segs(self, sample_id, chr_id, win_size=1000, min_r=0.1, stride_size=200,
                      online_feat_segs_fn=None, n_proc=16):
        """
        calculate feature map segments for given window

        :param sample_id:
        :param chr_id:
        :param win_size:
        :param min_r:
        :param stride_size:
        :param online_feat_segs_fn:
        :param n_proc: number of processor
        """

        self.sample_id = sample_id
        self.chr_id = chr_id

        # assure the feature segmentation list is empty before adding the feature maps
        self.chr_segs_unpredictable = []
        self.chr_segs_predictable = []
        self.chr_segs_predictable_feats = []

        ref_gap_chr = self.ref_gap_obj.loc[self.ref_gap_obj['CHROM'] == 'chr' + chr_id,
                                           ['START', 'END']] if not self.ref_gap_obj.empty else None

        # reference base
        logger.info('Loading reference sequence for sample {} chr: {}...'.format(sample_id, chr_id))
        self.rb_base_chr = self.ref_fa_obj.fetch('chr' + chr_id)
        chr_len = len(self.rb_base_chr)

        # reference mappabillity
        logger.info('Loading reference mappability for sample {} chr: {}...'.format(sample_id, chr_id))
        self.rb_mappability_chr = self.ref_bw_obj.values('chr' + chr_id, 0, chr_len)

        fil_pos = np.array([], dtype=np.int)
        if ref_gap_chr is not None:
            for _, i_row in ref_gap_chr.iterrows():
                # END is excluded
                fil_pos = np.concatenate((fil_pos, np.arange(i_row['START'], i_row['END'])))

        rb_base_pos = np.ones(chr_len, dtype=int)
        rb_base_pos[fil_pos] = 0

        # call pysam pileup() to read the whole chr
        logger.info('loading whole bam pileup for sample {} chr: {}...'.format(sample_id, chr_id))
        assert self.bam_obj_whole is not None
        whole_chr_pileup = self.bam_obj_whole.pileup('chr' + chr_id, stepper='nofilter', min_base_quality=0)

        # get the feature map for the whole chr
        # take almost 25 minutes for  chr1
        logger.info('calculating feature maps for whole chromosome, sample {}, chr {}...'.format(sample_id, chr_id))
        logger.info('>>>>>>>>>>this processing will take a few minutes (almost 25 minutes for chr 1)...')
        self.chr_feat_mat_whole = gen_feat_whole_chr(whole_chr_pileup,
                                                     self.rb_base_chr,
                                                     chr_len,
                                                     self.rb_mappability_chr)
        del whole_chr_pileup
        del self.rb_base_chr
        del self.rb_mappability_chr

        # self.chr_feat_mat_whole = np.zeros((self.n_features, chr_len))
        # logger.info('submitting through ThreadPoolExecutor, worker={}...'.format(n_proc))
        # with concurrent.futures.ThreadPoolExecutor(max_workers=n_proc) as pool:
        #     # There must use ThreadPoolExecutor. it is not working if using ProcessPoolExecutor
        #     res = [pool.submit(gen_feat_single, pileup_column) for pileup_column in whole_chr_pileup]
        #     for i_re in concurrent.futures.as_completed(res):
        #         i_ref_pos, *i_re_tup = i_re.result()
        #
        #         # if i_ref_pos % 1000000 == 0:
        #         # logger.info('feature generation at position {}'.format(i_ref_pos))
        #         # base coverage
        #         self.chr_feat_mat_whole[0, i_ref_pos] = i_re_tup[0]
        #         # base quality
        #         self.chr_feat_mat_whole[1, i_ref_pos] = i_re_tup[1]
        #         # base map quality
        #         self.chr_feat_mat_whole[2, i_ref_pos] = i_re_tup[2]
        #         # base_gc_cnt
        #         self.chr_feat_mat_whole[3, i_ref_pos] = i_re_tup[3]
        #         # base_a_cnt
        #         self.chr_feat_mat_whole[4, i_ref_pos] = i_re_tup[4]
        #         # base_t_cnt
        #         self.chr_feat_mat_whole[5, i_ref_pos] = i_re_tup[5]
        #         # base_c_cnt
        #         self.chr_feat_mat_whole[6, i_ref_pos] = i_re_tup[6]
        #         # base_g_cnt
        #         self.chr_feat_mat_whole[7, i_ref_pos] = i_re_tup[7]
        #
        #         # reference mappability
        #         self.chr_feat_mat_whole[8, i_ref_pos] = self.rb_mappability_chr[i_ref_pos]
        #
        #         self.chr_feat_mat_whole[9, i_ref_pos] = 1 if self.rb_base_chr[i_ref_pos] == 'A' else 0
        #         self.chr_feat_mat_whole[10, i_ref_pos] = 1 if self.rb_base_chr[i_ref_pos] == 'T' else 0
        #         self.chr_feat_mat_whole[11, i_ref_pos] = 1 if self.rb_base_chr[i_ref_pos] == 'C' else 0
        #         self.chr_feat_mat_whole[12, i_ref_pos] = 1 if self.rb_base_chr[i_ref_pos] == 'G' else 0

        seg_values, seg_starts, seg_lengths = find_seg(rb_base_pos)
        assert len(seg_values) == len(seg_starts) == len(seg_lengths)

        logger.info('segmenting {} bp-long {} bp step feature maps sample {}, chr {}...'.format(
            win_size, stride_size, sample_id, chr_id))

        for val_idx, i_val in enumerate(seg_values):
            if i_val == 0:
                # gap region
                self.chr_segs_unpredictable.append(np.array([seg_starts[val_idx],
                                                             seg_starts[val_idx] + seg_lengths[val_idx],
                                                             seg_lengths[val_idx], 0]))
            else:
                i_seg_start = seg_starts[val_idx]
                i_seg_len = seg_lengths[val_idx]

                if i_seg_len >= win_size:
                    i_start_indices, remain_len = seq_slide(i_seg_len, win_size, stride_size)
                    for i in i_start_indices:
                        i_w_start = int(i+i_seg_start)
                        i_w_end = int(i_w_start + win_size)

                        self.__get_feats_region(i_w_start, i_w_end, win_size, min_r)

                    if remain_len > 0:
                        i_w_start = int(i_seg_len-win_size)
                        i_w_end = int(i_seg_len)
                        self.__get_feats_region(i_w_start, i_w_end, win_size, min_r)
                else:
                    self.__get_feats_region(i_seg_start, i_seg_start+i_seg_len, win_size, min_r)

        del self.chr_feat_mat_whole
        gc.collect()

        self.chr_segs_unpredictable = np.vstack(self.chr_segs_unpredictable)
        self.chr_segs_predictable = np.vstack(self.chr_segs_predictable)
        self.chr_segs_predictable_feats = np.array(self.chr_segs_predictable_feats)

        if online_feat_segs_fn:
            logger.info('saving segments of {} bp-long {} bp step feature maps... {}'.format(win_size,
                                                                                             stride_size,
                                                                                             online_feat_segs_fn))
            np.savez_compressed(online_feat_segs_fn,
                                chr_segs_unpredictable=self.chr_segs_unpredictable,
                                chr_segs_predictable=self.chr_segs_predictable,
                                chr_segs_predictable_feats=self.chr_segs_predictable_feats)

    def cnv_call(self, out_cnv_call_fn, online_feat_segs_in_fn=None):
        """

        :param out_cnv_call_fn:
        :param online_feat_segs_in_fn:
        :return:
        """

        if len(self.chr_segs_predictable_feats) == 0:
            if online_feat_segs_in_fn is not None and os.path.exists(online_feat_segs_in_fn):
                logger.info('loading features from saved feature file: {}'.format(online_feat_segs_in_fn))
                with np.load(online_feat_segs_in_fn) as f_feat_seg_chr:
                    self.chr_segs_unpredictable = f_feat_seg_chr['chr_segs_unpredictable']
                    self.chr_segs_predictable = f_feat_seg_chr['chr_segs_predictable']
                    self.chr_segs_predictable_feats = f_feat_seg_chr['chr_segs_predictable_feats']
            else:
                raise ValueError('chromosome feature maps segments None.')

        if os.path.exists(out_cnv_call_fn):
            os.remove(out_cnv_call_fn)

        # prediction by model and output results
        logger.info('CNV calling...')
        assert self.model is not None
        y_prob = self.model.predict(self.chr_segs_predictable_feats, batch_size=self.batch_size)
        ypred_cls = y_prob.argmax(axis=-1)
        re_unpred = np.concatenate([self.chr_segs_predictable,
                                    np.full((self.chr_segs_predictable.shape[0], y_prob.shape[-1]+1), -1)], axis=-1)
        re_pred = np.concatenate([self.chr_segs_predictable, y_prob, ypred_cls.reshape([-1, 1])], axis=-1)

        re_whole = np.concatenate([re_unpred, re_pred])
        del re_unpred, re_pred

        logger.info('sorting the predict results by start point of each segment...')
        res = re_whole[re_whole[:, 0].argsort()]

        logger.info('writing the results into the file...')
        res_df = pd.DataFrame(res, columns=['start', 'end', 'length', 'pred_type',
                                            'pred_p_neu', 'pred_p_del', 'pred_p_dup', 'pred_cls'])
        res_df.to_csv(out_cnv_call_fn, sep='\t', index=False)
        logger.info('Done, predicting result saved at {}'.format(out_cnv_call_fn))

    def __load_model(self, model_weight_fn):

        logger.info('loading model weight...')
        assert os.path.exists(model_weight_fn)
        model_weight_name = os.path.splitext(os.path.basename(model_weight_fn))[0]
        model_in_lst = model_weight_name.split('-')
        model_name = model_in_lst[1]
        model_params_lst = re.findall(r"[-+]?\d*\.\d+|\d+", model_in_lst[0])

        logging.info('model name: {0}, model params(batch, epoch, lr, drop, fc, block, win): {1}'.format(
            model_name, model_params_lst))
        assert len(model_params_lst) >= 6

        batch_size = int(model_params_lst[0])
        drop = float(model_params_lst[3])
        fc_size = int(model_params_lst[4])
        blocks = (int(x) for x in model_params_lst[5])
        win_size = int(model_params_lst[6])

        model = None
        if model_name == 'cnvnet':
            model = cnv_net(win_size, self.n_features, self.n_classes,
                            drop=drop, blocks=blocks, fc_size=fc_size)

        model.load_weights(model_weight_fn)
        return model, batch_size

    def __get_feats_region(self, reg_start, reg_end, win_size, min_r):
        """

        :param reg_start:
        :param reg_end:
        :param win_size:
        :param min_r:
        :return:
        """
        i_feat_mat = self.chr_feat_mat_whole[:, reg_start:reg_end]
        i_base_cov = i_feat_mat[0]
        valide_base_cov = i_base_cov[i_base_cov > 0]
        if len(valide_base_cov) == 0:
            # no reads cover
            self.chr_segs_unpredictable.append(np.array([reg_start,
                                                         reg_end,
                                                         win_size, 1]))
        elif len(valide_base_cov) < win_size * min_r:
            # the number of reads coverage is not enough
            self.chr_segs_unpredictable.append(np.array([reg_start,
                                                         reg_end,
                                                         win_size, 2]))
        else:
            # ok, requirements satisfied
            # normalize
            i_x_max = np.max(i_feat_mat, axis=-1)
            i_x_max[i_x_max == 0] = 1
            i_feat_mat = i_feat_mat * 1.0 / i_x_max.reshape(len(i_x_max), 1)
            i_feat_mat = np.transpose(i_feat_mat)

            self.chr_segs_predictable.append(np.array([reg_start,
                                                       reg_end,
                                                       win_size, 3]))

            self.chr_segs_predictable_feats.append(i_feat_mat)



