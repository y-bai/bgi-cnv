#!/usr/bin/env sh

# =====================================================
# Description:
#
# =====================================================
#
# Created by YongBai on 2020/1/16 9:38 AM.

win=10000
step_size=2000
min_r=0.1

#10000bp's segment path for online call
input_root_dir="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/online_calling_features/multi_bin_segment_feature_for_252_0.1X_NIPT_like_1000genome_testsample"
input_win_root_dir="${input_root_dir}/segment_features_winsize${win}_stepsize${step_size}"

ae_model_dir="/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/model123"

#sample_ids=(HG01777 NA18602 HG03410 HG00533 HG01951 HG03713 HG02407 NA19059 HG02010 NA19076)
#sample_ids=(HG01777)
sample_ids=(HG01777 NA18602 HG03410 HG00533 HG01951 HG03713 HG02407 NA19059 HG02010 NA19076 HG00102 HG00109 HG00114 HG00120 HG00122 HG00130 HG00138 HG00158 HG00160 HG00176 HG00246 HG00251 HG00253 HG00257 HG00263 HG00264 HG00288 HG00306 HG00321)
chr_ids=($(echo {1..22}))
#chr_ids=("1")

for sample_id in "${sample_ids[@]}"
do
    for chr_id in "${chr_ids[@]}"
    do
        sleep 20s
        in_dir="${input_win_root_dir}/${sample_id}/data"
        # here out_dir = in_dir
        echo "python ../online_CNV_AE_gen_feat.py \
        -i ${in_dir} \
        -w ${win} \
        -p ${step_size} \
        -s ${sample_id} \
        -c ${chr_id} \
        -r ${min_r} \
        -m ${ae_model_dir}">OFG${win}_${sample_id}_${chr_id}.sh
        qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=40g,p=1 OFG${win}_${sample_id}_${chr_id}.sh
    done
done



#echo "python model_AE_gen_feat.py -w ${win} -d ${datat}"> AE${win}_model_feat_15-1.sh
#qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e15-1 -l vf=40g,p=1 AE${win}_model_feat_15-1.sh