#!/usr/bin/env bash

# =====================================================
# Description: The pipeline for calling CNV using deep learning model
# STEP 2: select CNV types (CNV, DUP, DEL) from parsed vcf file
# =====================================================
#
# Created by Yong Bai on 2019/8/13 11:39 AM.

# use the default arguments
echo "python ../data_prepare/vcfcnvselect_main.py"> run2_f.sh

qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e15-1 -l vf=32g,p=1 run2_f.sh