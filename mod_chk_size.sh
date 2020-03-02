#!/usr/bin/env bash

# =====================================================
# Description:
#
# =====================================================
#
# Created by Yong Bai on 2019/11/14 5:29 PM.


WIN=5000

ROOT_DIR="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data/"
IN_OUT_DIR=${ROOT_DIR}win_${WIN}
IN_FN="trains_mr0.10_mf0.01_original.h5"


echo "python mod_h5_chunk_size.py \
-f ${IN_FN} \
-d ${IN_OUT_DIR} "> MOD_R${WIN}_st_q_run8_9_val.sh
qsub -cwd -P P18Z10200N0124 -q st.q -l vf=70g,p=1  MOD_R${WIN}_st_q_run8_9_val.sh


