#!/usr/bin/env bash

# =====================================================
# Description: The pipeline for calling CNV using deep learning model
# STEP 5: 1kGP downsample data(0.1X, kmer=35) to
#              create features matrix for neu regions
# =====================================================
#
# Created by Yong Bai on 2019/7/30 1:35 PM.

BAM_DIR="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/KGP_remain_fq_bam_file/final_NIPT_like_bam"
CNV_Label_DIR="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/1k_cnvs_lab_feat"

# feature matrix before windowing.
# NOTE: this OUT_DIR is for saving low-pass data(NIPT-like, ie, after down-sampling)
OUT_DIR="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/true_NEU_feature_matrix"

REF_FASTA="/zfssz6/ST_MCHRI/BIGDATA/database/BGI-seq500_OSS_download/human_reference/hg38/Homo_sapiens_assembly38.fasta"
REF_MAP_BW="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/NIPT_35bp_hg38_map_gc/Homo_sapiens_assembly38.fasta.kmer35.bw"

SAMPLE_LIST_FNAME="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/sample.list"

for sample_id in `less ${SAMPLE_LIST_FNAME} |awk '{print $1}'`
do
    ds_bam_fn=${BAM_DIR}/${sample_id}/*.bam
    cnv_lab_fn=${CNV_Label_DIR}/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.${sample_id}.cnvs.labels

    echo "python ../data_prepare/mat_neu_data_prepare.py \
            -s ${sample_id} \
            -b ${ds_bam_fn} \
            -c ${cnv_lab_fn} \
            -o ${OUT_DIR} \
            -f ${REF_FASTA} \
            -m ${REF_MAP_BW}" > ${sample_id}_run5_f.sh
    qsub -cwd -P P18Z10200N0124 -q st.q -l vf=20g,p=1 ${sample_id}_run5_f.sh
done