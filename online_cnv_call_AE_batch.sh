#!/usr/bin/env bash

# =====================================================
# Description:
# This is onlince cnv call integreated with AE dimension
# reduction.
#
# =====================================================
#
# Created by Yong Bai on 2019/10/8 2:52 PM.


WIN_SIZE=10000
STEP_SIZE=2000

MODEL_ROOT_DIR="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out"
# online for 1KGP test sample
CNV_OUT_ROOT_DIR="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/1KGP/win_${WIN_SIZE}"

SEGS_IN_ROOT_DIR='/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/online_calling_features/multi_bin_segment_feature_for_252_0.1X_NIPT_like_1000genome_testsample'
SEGS_IN_DIR="${SEGS_IN_ROOT_DIR}/segment_features_winsize${WIN_SIZE}_stepsize${STEP_SIZE}/"

MIN_RATIO=0.1
N_FEATURE=9
N_CLASSES=3
#MODEL_TYPE='1:1:1'
MODEL_TYPE='1:2:3'

# model parameter, see model_train_main3.py
EPOCHS=100
BATCH_SIZE=128
DROP_OUT=0.5
LR=0.001
FC_SIZE=64
BLOCKS='4_3'
L2R=0.0001
TEMPERATURE=6
LBL_SMT_F=0

PW=-8 # 111->221
#SE='25265635-25335230' #SE='25265635-25335230'
SE='a'
N_PROC=4  # DO NOT SET N_PROC>4, OTHERWISE GPU (~15.8G) WILL BE OUT OF MEMORY.

#sample_ids=(HG01777 NA18602 HG03410 HG00533 HG01951 HG03713 HG02407 NA19059 HG02010 NA19076)
sample_ids=(HG01777 NA18602 HG03410 HG00533 HG01951 HG03713 HG02407 NA19059 HG02010 NA19076 HG00102 HG00109 HG00114 HG00120 HG00122 HG00130 HG00138 HG00158 HG00160 HG00176 HG00246 HG00251 HG00253 HG00257 HG00263 HG00264 HG00288 HG00306 HG00321)
chr_ids=($(echo {1..22}))

for SAMPLE_ID in "${sample_ids[@]}"
do
    for CHR_ID in "${chr_ids[@]}"
    do
        sleep 30s
        echo "python ../online_cnv_call_AE_main.py \
-s ${SAMPLE_ID} \
-c ${CHR_ID} \
-r ${MIN_RATIO} \
-w ${WIN_SIZE} \
-i ${SEGS_IN_DIR} \
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
-q ${LBL_SMT_F}" > OC${SAMPLE_ID}_${CHR_ID}.sh
qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=40g,p=1 OC${SAMPLE_ID}_${CHR_ID}.sh

    done
done

# pid=9841