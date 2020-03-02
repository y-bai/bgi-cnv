#!/usr/bin/env bash

# =====================================================
# Description: The pipeline for calling CNV using deep learning model
# STEP 8: svae training dataset for DUP and DEL as npz
# =====================================================
#
# Created by Yong Bai on 2019/8/5 11:10 AM.

#SAMPLE_LIST_FN="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/sample.list"
#IN_DATA_DIR="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/1k_ds_trains"
#OUT_DIR="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/1dcnn_resnet"
#
#
#for bam in `less ${SAMPLE_LIST_FN} |awk '{print $1}'`
#do
#    sample_id=${bam}
#    echo "python ../data_prepare/model_crt_train_test_npz.py -s ${sample_id} -i ${IN_DATA_DIR} -o ${OUT_DIR}"> ${sample_id}_run8.sh
#
#    qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=8g,p=1 ${sample_id}_run8.sh
#done

win=2000
ratio=0.1
cnv_min_f=0.01
neu_min_f=-2
normalize='False'
n_cpu=20

TRAIN_TEST_FN="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data/train_test0.10_sample_ids.npz"
#IN_DATA_DIR="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/multi_bin_ds_trains_feature/${win}_ds_trains"
IN_CNV_DATA_DIR="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/multi_bin_ds_trains_feature"
IN_NEU_DATA_DIR="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/multi_bin_ds_neus_trains_feature"
OUT_ROOT_DIR="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data"

echo "python ../data_prepare/model_crt_dataset4generator2.py \
-f ${TRAIN_TEST_FN} \
-w ${win} \
-r ${ratio} \
-q ${cnv_min_f} \
-p ${neu_min_f} \
-n ${normalize} \
-v ${IN_CNV_DATA_DIR} \
-u ${IN_NEU_DATA_DIR} \
-t ${n_cpu} \
-o ${OUT_ROOT_DIR}"> R${win}_st_q_run8_9_h5.sh
qsub -cwd -P P18Z10200N0124 -q st.q -l vf=70g,p=20 -binding linear:20 R${win}_st_q_run8_9_h5.sh
#qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e15-1 -l vf=40g,p=1 R${win}_15-1_run8_9_h5.sh
