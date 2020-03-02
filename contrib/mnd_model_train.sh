#!/usr/bin/env bash

# =====================================================
# Description:
#
# =====================================================
#
# Created by Yong Bai on 2019/8/29 9:33 AM.

echo "ulimit -l; horovodrun -np 1 \
    --start-timeout 300 \
    --mpi-threads-disable 1 \
    python model_train_main.py"> mnd_run.sh

qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e16-1 -l vf=40g,p=1,g=1 mnd_run.sh