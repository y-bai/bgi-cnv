#!/usr/bin/env bash

# =====================================================
# Description: The pipeline for calling CNV using deep learning model
# STEP 20: online data generation
# =====================================================
#
# Created by Yong Bai on 2019/9/3 10:49 AM.

# CODE_BASE_DIR="/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/cnv_detect_final"

BASE_OUT_DIR="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online"

BAM_DIR="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/KGP_remain_fq_bam_file/final_NIPT_like_bam"
REF_FA_F="/zfssz6/ST_MCHRI/BIGDATA/database/BGI-seq500_OSS_download/human_reference/hg38/Homo_sapiens_assembly38.fasta"
REF_MAP_BW="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/NIPT_35bp_hg38_map_gc/Homo_sapiens_assembly38.fasta.kmer35.bw"

MODEL_ROOT_DIR="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/model"
MODEL_WEIGHT_FN_TMP="${MODEL_ROOT_DIR}/b1024_e64_lr0.001_dr0.5_fc64_blk443_win1000-cnvnet.hdf5"

MIN_RATIO=0.1
WIN_SIZE=1000
CHR_ID="1"
STEP_SIZE=200
N_FEATURE=13
N_CLASSES=3
N_PROC=32

SAMPLE_ID="NA12878"
bam_fn=${BAM_DIR}/${SAMPLE_ID}/*.bam

echo "python ../online/online_predict_end2end.py \
-s ${SAMPLE_ID} \
-b ${bam_fn} \
-r ${MIN_RATIO} \
-w ${WIN_SIZE} \
-c ${CHR_ID} \
-o ${BASE_OUT_DIR} \
-p ${STEP_SIZE} \
-d ${MODEL_WEIGHT_FN_TMP} \
-u ${N_FEATURE} \
-l ${N_CLASSES} \
-t ${N_PROC} \
-f ${REF_FA_F} \
-m ${REF_MAP_BW}" > ${SAMPLE_ID}_run30_e15-1.sh
qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e15-1 -l vf=40g,p=1 ${SAMPLE_ID}_run30_e15-1.sh
#qsub -cwd -P P18Z10200N0124 -q st.q -l vf=40g,p=1 ${SAMPLE_ID}_run30_f.sh