#!/usr/bin/env bash

# =====================================================
# Description:
#
# =====================================================
#
# Created by Yong Bai on 2019/11/11 1:51 PM.

win=1000
echo "python model_train_main2.py \
-w ${win}"> W${win}_model_train2_temp6.sh
qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e16-2 -l vf=40g,p=1 W${win}_model_train2_temp6.sh

# e15-1 10k