#!/usr/bin/env bash

# =====================================================
# Description:
#
# =====================================================
#
# Created by Yong Bai on 2019/11/19 4:35 PM.

win=1000
step_size=200
n_cpus=10
sample_id='NA12878'
chr_id='1'

out_type='eval' # repo

in_fn=win${win}_step${step_size}_r0.10_chr${chr_id}-cnv-call-a-result-mp-gpu-pw-8.csv
in_dir=/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/${sample_id}/cnv_call
out_dir=/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/${sample_id}/cnv_call

echo "python prediction_merge.py \
-w ${win} \
-t ${n_cpus} \
-s ${step_size} \
-m ${sample_id} \
-c ${chr_id} \
-p ${out_type} \
-f ${in_fn} \
-i ${in_dir} \
-o ${out_dir}">Merge_${win}_${sample_id}_${chr_id}.sh
#qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e16-1 -l vf=40g,p=1 Merge_${win}_${sample_id}_${chr_id}.sh
qsub -cwd -P P18Z10200N0124 -q st.q -l vf=20g,p=10 -binding linear:10 Merge_${win}_${sample_id}_${chr_id}.sh
