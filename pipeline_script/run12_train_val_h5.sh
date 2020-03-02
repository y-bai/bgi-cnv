#!/usr/bin/env bash

# =====================================================
# Description:
# STEP 12: split "train data samples" to trains samples
# and validation samples.
# =====================================================
#
# Created by Yong Bai on 2019/9/19 1:40 AM.

win=5000
ratio=0.1
min_f=0.01
train_rate=0.8
nb_cls_prop='1:1:1'
n_cpu=30

DATA_ROOT_DIR="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data"


echo "python ../dataset_train/train_val_dataset2.py \
-w ${win} \
-r ${ratio} \
-q ${min_f} \
-t ${train_rate} \
-b ${nb_cls_prop} \
-p ${n_cpu} \
-i ${DATA_ROOT_DIR} \
-o ${DATA_ROOT_DIR}" > R${win}_train_val_dataset_st_q_h5.sh
qsub -cwd -P P18Z10200N0124 -q st.q -l vf=70g,p=30 -binding linear:30 R${win}_train_val_dataset_st_q_h5.sh
#qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e16-1 -l vf=50g,p=1 R${win}_train_val_dataset_e16-1_h5.sh