#!/usr/bin/env bash

# =====================================================
# Description:
#
# =====================================================
#
# Created by Yong Bai on 2019/10/8 2:52 PM.

MODEL_ROOT_DIR="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out"
CNV_OUT_ROOT_DIR='/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online'
#SEGS_IN_ROOT_DIR='/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online'
SEGS_IN_ROOT_DIR='/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/online_calling_features/252_0.1X_NIPT_like_1000genome_testsample_online_calling_features/'

MIN_RATIO=0.1
WIN_SIZE=1000
STEP_SIZE=200
N_FEATURE=13
N_CLASSES=3

SAMPLE_ID='NA12878'
#SAMPLE_ID='18B3467030'
CHR_ID="4"
MODEL_TYPE='1:1:1'

EPOCHS=100   # 64->50
BATCH_SIZE=512  # 1024->512
DROP_OUT=0.5  # 0.5 ->0.1
LR=0.001
FC_SIZE=64  # 64->32
BLOCKS='4_4_3'  # 4_4_3->4_4_1
L2R=0.0001
TEMPERATURE=6  # 5->
LBL_SMT_F=0  # 0.1->0


PW=-8 # 111->221
#SE='25265635-25335230' #SE='25265635-25335230'
SE='a'
N_PROC=4  # DO NOT SET N_PROC>4, OTHERWISE GPU (~15.8G) WILL BE OUT OF MEMORY.

#echo "/zfssz5/BC_PS/liuyu6/software/anaconda3/envs/ai/bin/python online_cnv_call_main.py \
echo "python online_cnv_call_main.py \
-s ${SAMPLE_ID} \
-c ${CHR_ID} \
-r ${MIN_RATIO} \
-w ${WIN_SIZE} \
-i ${SEGS_IN_ROOT_DIR} \
-o ${CNV_OUT_ROOT_DIR} \
-g ${MODEL_ROOT_DIR} \
-n ${N_FEATURE} \
-t ${N_CLASSES} \
-m ${MODEL_TYPE} \
-p ${STEP_SIZE} \
-e ${EPOCHS} \
-b ${BATCH_SIZE} \
-d ${DROP_OUT} \
-l ${LR} \
-f ${FC_SIZE} \
-x ${SE} \
-k ${BLOCKS} \
-u ${PW} \
-z ${N_PROC} \
-y ${L2R} \
-a ${TEMPERATURE} \
-q ${LBL_SMT_F}" > M${SAMPLE_ID}_${CHR_ID}_online_cnvcall_e15-1-8.sh
qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e15-1 -l vf=40g,p=1 M${SAMPLE_ID}_${CHR_ID}_online_cnvcall_e15-1-8.sh
# /zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/100W_sample_id.txt
#echo "python online_cnv_call_main.py \