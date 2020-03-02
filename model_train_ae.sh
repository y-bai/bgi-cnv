#!/usr/bin/env sh

# =====================================================
# Description:
#
# =====================================================
#
# Created by model_train_ae.sh on 2020/1/2 3:53 PM.

win=10000
echo "python model_train_main_AE.py -w ${win}"> W${win}_model_train_ae_16-2.sh
qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e16-2 -l vf=40g,p=1 W${win}_model_train_ae_16-2.sh