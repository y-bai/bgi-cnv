#!/usr/bin/env bash

# =====================================================
# Description:
#  whole feature map segmentation for given win and step size
# [online cnv call]
# =====================================================
#
# Created by Yong Bai on 2019/9/23 2:23 PM.

BASE_OUT_DIR="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online"

REF_FA_F="/zfssz6/ST_MCHRI/BIGDATA/database/BGI-seq500_OSS_download/human_reference/hg38/Homo_sapiens_assembly38.fasta"

MIN_RATIO=0.1
WIN_SIZE=1000
CHR_ID="1"
STEP_SIZE=1000 # 200->10, 200->35
N_PROC=16
N_FEATURES=13

SAMPLE_ID="NA12878"

#SE='25265635-25335230' #25266150|25334730
SE='a'
echo "python ../online_feat_gen/online_feat_seg_opt_main.py \
-s ${SAMPLE_ID} \
-r ${MIN_RATIO} \
-w ${WIN_SIZE} \
-c ${CHR_ID} \
-o ${BASE_OUT_DIR} \
-p ${STEP_SIZE} \
-t ${N_PROC} \
-u ${N_FEATURES} \
-x ${SE} \
-f ${REF_FA_F} " > ${SAMPLE_ID}_run21_ff.sh
#qsub -cwd -P P18Z10200N0124 -q st.q -l vf=40g,p=1 ${SAMPLE_ID}_run21_f.sh
qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e15-1 -l vf=40g,p=1 ${SAMPLE_ID}_run21_ff.sh