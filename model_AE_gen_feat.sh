#!/usr/bin/env sh

# =====================================================
# Description:
#
# =====================================================
#
# Created by model_AE_gen_feat on 2020/1/7 3:48 PM.

win=10000
datat="val"
echo "python model_AE_gen_feat.py -w ${win} -d ${datat}"> AE${win}_model_feat_15-1.sh
qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e15-1 -l vf=40g,p=1 AE${win}_model_feat_15-1.sh