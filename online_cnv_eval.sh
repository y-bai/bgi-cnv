#!/usr/bin/env bash

# =====================================================
# Description:
#
# =====================================================
#
# Created by Yong Bai on 2019/11/26 1:04 PM.


win=1000
step_size=200
n_cpus=20
sample_id='HG01777'
chr_id='1'

truth_cnv_dir=/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/1k_cnvs_lab_feat

in_fn=M${sample_id}_${chr_id}_${win}_${step_size}_out_cnv_eval.csv
in_dir=/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/1KGP/${sample_id}/cnv_call

out_dir=/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/1KGP/${sample_id}/cnv_call

echo "python online_cnv_call_eval.py \
-d ${truth_cnv_dir} \
-w ${win} \
-t ${n_cpus} \
-s ${step_size} \
-m ${sample_id} \
-c ${chr_id} \
-f ${in_fn} \
-i ${in_dir} \
-o ${out_dir}">Eval_${win}_${sample_id}_${chr_id}.sh
#qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e16-1 -l vf=40g,p=1 Eval_${win}_${sample_id}_${chr_id}.sh
qsub -cwd -P P18Z10200N0124 -q st.q -l vf=20g,p=20 -binding linear:20 Eval_${win}_${sample_id}_${chr_id}.sh