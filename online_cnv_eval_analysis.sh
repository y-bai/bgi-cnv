#!/usr/bin/env bash

# =====================================================
# Description:
#
# =====================================================
#
# Created by Yong Bai on 2019/11/28 9:15 AM.

win=10000
step_size=2000
n_cpus=10

#sample_ids=(HG01777 NA18602 HG03410 HG00533 HG01951 HG03713 HG02407 NA19059 HG02010 NA19076)
sample_ids=(HG01777 NA18602 HG03410 HG00533 HG01951 HG03713 HG02407 NA19059 HG02010 NA19076 HG00102 HG00109 HG00114 HG00120 HG00122 HG00130 HG00138 HG00158 HG00160 HG00176 HG00246 HG00251 HG00253 HG00257 HG00263 HG00264 HG00288 HG00306 HG00321)

#sample_ids=(HG00251)

for sample_id in "${sample_ids[@]}"
do
    sleep 10s
    in_dir=/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/1KGP/win_${win}/${sample_id}/cnv_call
    out_dir=/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/1KGP/win_${win}/${sample_id}/cnv_call
    echo "python ../online_cnv_eval_analysis.py \
-w ${win} \
-t ${n_cpus} \
-s ${step_size} \
-m ${sample_id} \
-i ${in_dir} \
-o ${out_dir}">Analysis_${win}_${sample_id}.sh
qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=4g,p=1 Analysis_${win}_${sample_id}.sh
#qsub -cwd -P P18Z10200N0124 -q st.q -l vf=10g,p=20 -binding linear:20 Analysis_${win}_${sample_id}.sh

done