#!/usr/bin/env bash

# =====================================================
# Description:
#
# =====================================================
#
# Created by Yong Bai on 2019/9/30 4:24 PM.

echo "python offline_test_data_partition.py"> test_data_partition_e16-1-run.sh
qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e16-1 -l vf=40g,p=1,g=1 test_data_partition_e16-1-run.sh