#!/usr/bin/env bash

# =====================================================
# Description: The pipeline for calling CNV using deep learning model
# STEP 6: create training dataset for DUP and DEL
# =====================================================
#
# Created by Yong Bai on 2019/8/13 1:17 PM.

# This is input dir for DEL or DUP feat
FEAT_DIR="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/true_CNV_feature_matrix"
SAMPLE_LIST_FNAME="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/sample.list"
# This is output dir for DEL or DUP
BASE_OUT_DIR="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/multi_bin_ds_trains_feature"

win_size=1000
min_ratio=0.1
min_f=0.01
in_feat_type=1

OUT_DIR="${BASE_OUT_DIR}/${win_size}_ds_trains"

for sample_id in `less ${SAMPLE_LIST_FNAME} |awk '{print $1}'`
do

    feat_fn=${FEAT_DIR}/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.${sample_id}.cnvs.features

    echo "python ../data_prepare/mat_crt_train_data.py \
            -s ${sample_id} \
            -f ${feat_fn} \
            -w ${win_size} \
            -q ${min_f} \
            -r ${min_ratio} \
            -o ${OUT_DIR} \
            -t ${in_feat_type} \
            -n False"> ${sample_id}_run6_f.sh
    qsub -cwd -P P18Z10200N0124 -q st.q -l vf=20g,p=1 ${sample_id}_run6_f.sh
done
