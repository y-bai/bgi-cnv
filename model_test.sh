#!/usr/bin/env bash

# =====================================================
# Description:
#
# =====================================================
#
# Created by Yong Bai on 2019/8/20 3:17 PM.


echo "python model_test_main.py"> model_test_e15-1-run.sh
qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e15-1 -l vf=40g,p=1,g=1 model_test_e15-1-run.sh