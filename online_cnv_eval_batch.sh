#!/usr/bin/env bash

# =====================================================
# Description:
#
# =====================================================
#
# Created by Yong Bai on 2019/11/26 1:04 PM.


win=10000
step_size=2000
n_cpus=10
#sample_ids=(HG01777 NA18602 HG03410 HG00533 HG01951 HG03713 HG02407 NA19059 HG02010 NA19076)
#sample_ids=(HG01777)
sample_ids=(HG01777 NA18602 HG03410 HG00533 HG01951 HG03713 HG02407 NA19059 HG02010 NA19076 HG00102 HG00109 HG00114 HG00120 HG00122 HG00130 HG00138 HG00158 HG00160 HG00176 HG00246 HG00251 HG00253 HG00257 HG00263 HG00264 HG00288 HG00306 HG00321)
chr_ids=($(echo {1..22}))
#chr_ids=("1")

truth_cnv_dir=/zfssz6/ST_MCHRI/BIGDATA/P18Z10200N0124/NIPT_CNV/NIPT_CNV/1k_cnvs_lab_feat

for sample_id in "${sample_ids[@]}"
do
    for chr_id in "${chr_ids[@]}"
    do
        sleep 10s
        in_fn=M${sample_id}_${chr_id}_${win}_${step_size}_out_cnv_eval.csv
        in_dir=/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/1KGP/win_${win}/${sample_id}/cnv_call
        out_dir=/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/1KGP/win_${win}/${sample_id}/cnv_call

        echo "python ../online_cnv_call_eval_main.py \
-d ${truth_cnv_dir} \
-w ${win} \
-t ${n_cpus} \
-s ${step_size} \
-m ${sample_id} \
-c ${chr_id} \
-f ${in_fn} \
-i ${in_dir} \
-o ${out_dir}">Eval_${win}_${sample_id}_${chr_id}.sh
        qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=10g,p=1 Eval_${win}_${sample_id}_${chr_id}.sh
#        qsub -cwd -P P18Z10200N0124 -q st.q -l vf=10g,p=${n_cpus} -binding linear:${n_cpus} Eval_${win}_${sample_id}_${chr_id}.sh

    done
done