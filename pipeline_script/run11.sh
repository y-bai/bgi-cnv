#!/usr/bin/env bash

# =====================================================
# Description:
# STEP 11: split "data samples" for trains and test "sample ids".
# =====================================================
#
# Created by Yong Bai on 2019/9/18 3:13 PM.


win=1000
ratio=0.1
min_f=0.01
test_split_rate=0.1

DATA_SAMPLE_ROOT_DIR="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data"
SAMPLE_LIST_FN="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/sample.list"


echo "python ../dataset_train/train_test_data.py \
-w ${win} \
-r ${ratio} \
-q ${min_f} \
-t ${test_split_rate} \
-i ${DATA_SAMPLE_ROOT_DIR} \
-l ${SAMPLE_LIST_FN}" > train_test_split_run11.sh

qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=8g,p=1 train_test_split_run11.sh