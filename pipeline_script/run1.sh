#!/usr/bin/env bash

# =====================================================
# Description: The pipeline for calling CNV using deep learning model
# STEP 1: parse cvf file
# =====================================================
#
# Created by Yong Bai on 2019/8/13 10:19 AM.

# here we use the default arguments
echo "python ../data_prepare/vcfreader_main.py"> run1_f.sh

qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e15-1 -l vf=40g,p=1 run1_f.sh