#!/usr/bin/env bash

# =====================================================
# Description:
#
# =====================================================
#
# Created by Yong Bai on 2019/8/20 3:17 PM.

win=1000
echo "python model_train_main.py"> W${win}_model_train1.sh
qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e15-1 -l vf=60g,p=1 W${win}_model_train1.sh

# batch_size=512
#echo "python model_train_main.py"> model_train_e16-1_np.sh
#qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e16-1 -l vf=100g,p=1 model_train_e16-1_np.sh

