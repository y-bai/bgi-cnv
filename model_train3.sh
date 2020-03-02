#!/usr/bin/env bash

# =====================================================
# Description:
#
# =====================================================
#
# Created by Yong Bai on 2019/11/11 1:51 PM.

win=10000
echo "python model_train_main3.py -w ${win}"> Tr${win}_model_train3_temp6.sh
qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e15-1 -l vf=40g,p=1 Tr${win}_model_train3_temp6.sh
