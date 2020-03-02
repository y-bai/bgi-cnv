#!/usr/bin/env bash

# =====================================================
# Description:
#
# =====================================================
#
# Created by Yong Bai on 2019/11/14 5:29 PM.

WIN=10000

ROOT_DIR="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data/"
IN_OUT_DIR=${ROOT_DIR}win_${WIN}
IN_TRAIN_FN="trains_mr0.10_mf0.01_train_111.h5"
IN_VAL_FN="trains_mr0.10_mf0.01_val_111.h5"


echo "python merge_train_val_data_h5.py \
-t ${IN_TRAIN_FN} \
-v ${IN_VAL_FN} \
-d ${IN_OUT_DIR} "> MG_R${WIN}.sh
qsub -cwd -P P18Z10200N0124 -q st.q -l vf=70g,p=1  MG_R${WIN}.sh


