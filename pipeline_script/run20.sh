#!/usr/bin/env bash

# =====================================================
# Description:
#  generate whole feature map for given sample and chromosome
# [online cnv call]
# =====================================================
#
# Created by Yong Bai on 2019/9/23 11:36 AM.

BASE_OUT_DIR="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online"

BAM_DIR="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/KGP_remain_fq_bam_file/final_NIPT_like_bam"
REF_FA_F="/zfssz6/ST_MCHRI/BIGDATA/database/BGI-seq500_OSS_download/human_reference/hg38/Homo_sapiens_assembly38.fasta"
REF_MAP_BW="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/NIPT_35bp_hg38_map_gc/Homo_sapiens_assembly38.fasta.kmer35.bw"

CHR_ID="1"
SAMPLE_ID="NA12878"  # HG00740 NA12878
bam_fn=${BAM_DIR}/${SAMPLE_ID}/*.bam
N_FEATURE=13
N_PROC=48
FMT="bam"

echo "python ../online_feat_gen/online_feat_gen_whole_chr_main.py \
-s ${SAMPLE_ID} \
-b ${bam_fn} \
-c ${CHR_ID} \
-a ${FMT} \
-u ${N_FEATURE} \
-t ${N_PROC} \
-o ${BASE_OUT_DIR} \
-f ${REF_FA_F} \
-m ${REF_MAP_BW}" > ${SAMPLE_ID}_run20_e15-1.sh
#qsub -cwd -P P18Z10200N0124 -q st.q -l vf=40g,p=1 ${SAMPLE_ID}_run20_test.sh

qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e15-1 -l vf=40g,p=1 ${SAMPLE_ID}_run20_e15-1.sh