#!/usr/bin/env bash

# =====================================================
# Description: The pipeline for calling CNV using deep learning model
# STEP 9: save training dataset for NEU as npz
# =====================================================
#
# Created by Yong Bai on 2019/8/13 1:58 PM.


win=1000
ratio=0.1
neu_f=-2
normalize='False'

SAMPLE_LIST_FN="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/sample.list"
IN_DATA_DIR="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/multi_bin_ds_neus_trains_feature/${win}_ds_neus_trains"
OUT_DIR="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/data"


#for bam in `less ${SAMPLE_LIST_FN} |awk '{print $1}'`
#do
#    sample_id=${bam}
#    echo "python ../data_prepare/model_crt_train_test_npz.py -s ${sample_id} -i ${IN_DATA_DIR} -q ${neu_f} -o ${OUT_DIR}"> ${sample_id}_run9.sh
#
#    qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=8g,p=1 ${sample_id}_run9.sh
#done

for bam in `less ${SAMPLE_LIST_FN} |awk '{print $1}'`
do
    sample_id=${bam}
    echo "python ../data_prepare/model_crt_dataset4generator.py \
-s ${sample_id} \
-w ${win} \
-r ${ratio} \
-q ${neu_f} \
-n ${normalize} \
-i ${IN_DATA_DIR} \
-o ${OUT_DIR}"> ../run/${sample_id}_run9.sh

    qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=8g,p=1 ../run/${sample_id}_run9.sh
done

