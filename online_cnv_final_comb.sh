#!/usr/bin/env bash

# =====================================================
# Description:
#
# =====================================================
#
# Created by Yong Bai on 2019/12/16 10:46 PM.

win_size=10000
step_size=2000
sample_ids="HG01777 NA18602 HG03410 HG00533 HG01951 HG03713 HG02407 NA19059 HG02010 NA19076 HG00102 HG00109 HG00114 HG00120 HG00122 HG00130 HG00138 HG00158 HG00160 HG00176 HG00246 HG00251 HG00253 HG00257 HG00263 HG00264 HG00288 HG00306 HG00321"
reg_size="2000 10000 20000 50000 100000"

echo "python online_cnv_final_comb.py -s '${sample_ids}' -r '${reg_size}' -w ${win_size} -p ${step_size}">COMB_online_cnv_final.sh

qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=10g,p=1 COMB_online_cnv_final.sh