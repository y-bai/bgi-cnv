#!/usr/bin/env bash

# =====================================================
# Description: The pipeline for calling CNV using deep learning model
# STEP 4: create feature matrix for CNVs (DUP, DEL)
# =====================================================
#
# Created by Yong Bai on 2019/8/13 12:29 PM.

BAM_DIR="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/KGP_remain_fq_bam_file/final_NIPT_like_bam"
SAMPLE_LIST_FNAME="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/sample.list"
REF_FA_F="/zfssz6/ST_MCHRI/BIGDATA/database/BGI-seq500_OSS_download/human_reference/hg38/Homo_sapiens_assembly38.fasta"
REF_MAP_BW="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/NIPT_35bp_hg38_map_gc/Homo_sapiens_assembly38.fasta.kmer35.bw"
CNV_LABEL_DIR="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/1k_cnvs_lab_feat"

# feature matrix before windowing.
# NOTE: this OUT_DIR is for saving low-pass data(NIPT-like, ie, after down-sampling)
OUT_DIR="/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/true_CNV_feature_matrix"

for sample_id in `less ${SAMPLE_LIST_FNAME} |awk '{print $1}'`
do
    bam_fn=${BAM_DIR}/${sample_id}/*.bam
    cnv_label_fn=${CNV_LABEL_DIR}/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.${sample_id}.cnvs.labels
    echo "python ../data_prepare/mat_cnvs_data_prepare.py \
            --sample_id ${sample_id} \
            --bam_fn ${bam_fn} \
            --cnv_labels_fn ${cnv_label_fn} \
            --out_dir ${OUT_DIR}
            --ref_fa_f ${REF_FA_F} \
            --ref_map_f ${REF_MAP_BW}"> ${sample_id}_run4_f.sh

    qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=32g,p=1 ${sample_id}_run4_f.sh
done