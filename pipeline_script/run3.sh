#!/usr/bin/env bash

# =====================================================
# Description: The pipeline for calling CNV using deep learning model
# STEP 3: Create CNV regions for each sample (1000 GP)
# =====================================================
#
# Created by Yong Bai on 2019/8/13 12:09 PM.

SAMPLE_LIST_FNAME="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/sample.list"

echo "python ../data_prepare/labeled_cnvs_preprocess.py \
        --n_p 40 \
        --f_samseqid_mapping ${SAMPLE_LIST_FNAME}" > run3_f.sh

qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e15-1 -l vf=32g,p=1 run3_f.sh