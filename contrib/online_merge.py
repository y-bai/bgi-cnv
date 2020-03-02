#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: online_merge.py
    Description:
    
Created by Yong Bai on 2019/10/21 2:53 PM.
"""

import os
import numpy as np
import pandas as pd

import pyhsmm
import pybasicbayes
import pyhsmm.basic.distributions as distributions
from pyhsmm.util.text import progprint_xrange
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import copy, os


sample_id = 'NA12878'
chr_id = '1'
online_candidate_res_root_dir='/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/'+sample_id+'/cnv_call'

online_re_fname = 'win1000_step35_r0.10_chr1-cnv-call-a-result.csv'
# online_re_fname = 'win1000_step200_r0.10_chr1-cnv-call-result.csv'
# online_re_fname = 'win1000_step10_r0.10_chr1-cnv-call-p-85514777-85540460-result.csv'
# online_re_fname = 'win1000_step35_r0.10_chr1-cnv-call-p-85514777-85540460-result.csv'

online_fn = os.path.join(online_candidate_res_root_dir, online_re_fname)

online_df = pd.read_csv(online_fn, sep='\t')

obs_dim = 2
Nmax = 5

obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.3,
                'nu_0':obs_dim+10
                }
# obs_hypparams = {'h_0':np.zeros(obs_dim),
#             'J_0':np.ones(obs_dim) * 0.001, #sq_0 #changes the hidden state detection (the lower the better) #0.001
#             'alpha_0':np.ones(obs_dim) * 0.1, #(make the number of hidden states worse higher the better)
#             'beta_0':np.ones(obs_dim) * 1}

dur_hypparams = {'alpha_0':2*30*500,
                 'beta_0':3}

obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

datas = online_df.loc[online_df['indicator']==3, ['p_del','p_dup']].values

posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
        # alpha=6.,gamma=6., # better to sample over these; see concentration-resampling.py 10,1
        alpha=1.,gamma=1./4,
        init_state_concentration=600., # pretty inconsequential
        obs_distns=obs_distns,
        dur_distns=dur_distns)
posteriormodel.add_data(datas,trunc=80)
for idx in progprint_xrange(40): # 100->50
    posteriormodel.resample_model()#num_procs=1)

exp_state = np.empty(len(online_df), dtype=int)
indicators = online_df['indicator'].values
exp_state[indicators==3] = posteriormodel.stateseqs[0]
online_df['exp_state']=exp_state
online_df.to_csv('/zfssz2/ST_MCHRI/BIGDATA/PROJECT/NIPT_CNV/f_cnv_out/online/NA12878/out1.csv', sep='\t')



