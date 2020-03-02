#!/usr/bin/env bash

# =====================================================
# Description:
#
# =====================================================
#
# Created by Yong Bai on 2019/11/19 4:35 PM.


win=1000
step_size=200
n_cpus=20
out_type='eval' # repo


sample_ids=(NA18602 HG03410 HG00533 HG01951 HG03713 HG02407 NA19059 HG02010 NA19076)
#sample_ids=(HG01777)

chr_ids=($(echo {1..22}))

truth_cnv_dir=/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/1k_cnvs_lab_feat

for sample_id in "${sample_ids[@]}"
do
    for chr_id in "${chr_ids[@]}"
    do
        sleep 10s
        in_fn=win${win}_step${step_size}_r0.10_chr${chr_id}-cnv-call-a-result-mp-gpu-pw-8.csv
        in_dir=/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/1KGP_bak/${sample_id}/cnv_call
        out_dir=/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/1KGP_bak/${sample_id}/cnv_call

        echo "python ../prediction_merge.py \
-w ${win} \
-t ${n_cpus} \
-s ${step_size} \
-m ${sample_id} \
-c ${chr_id} \
-p ${out_type} \
-f ${in_fn} \
-i ${in_dir} \
-o ${out_dir}">Merge_${win}_${sample_id}_${chr_id}.sh

        qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=1g,p=1 Merge_${win}_${sample_id}_${chr_id}.sh
#        qsub -cwd -P P18Z10200N0124 -q st.q -l vf=10g,p=20 -binding linear:20 Merge_${win}_${sample_id}_${chr_id}.sh
    done
done









