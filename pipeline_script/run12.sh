#!/usr/bin/env bash

# =====================================================
# Description:
# STEP 12: split "train data samples" to trains samples
# and validation samples.
# =====================================================
#
# Created by Yong Bai on 2019/9/19 1:40 AM.

win=1000
ratio=0.1
min_f=0.01
train_rate=0.8
nb_cls_prop='1:2:3'

DATA_ROOT_DIR="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data"


echo "python ../dataset_train/train_val_dataset.py \
-w ${win} \
-r ${ratio} \
-q ${min_f} \
-t ${train_rate} \
-b ${nb_cls_prop} \
-i ${DATA_ROOT_DIR} \
-o ${DATA_ROOT_DIR}" > train_val_dataset_run12.sh

qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=40g,p=1 train_val_dataset_run12.sh