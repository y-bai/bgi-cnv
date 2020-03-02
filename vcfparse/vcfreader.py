'''
The class to parse vcf file to get meta-information and data
Tested on sv vcf file

@author: Yong Bai, created on May 21, 2019
'''

import numpy as np
import pandas as pd
import multiprocessing
from itertools import zip_longest
from itertools import repeat
import time 
import os

import vcf


def parse_data(rec,args):
    '''
    function to read a line data
    '''
    # init a dict
    
    if rec is None:
        return None
    
    (sv_header,info_ids,sample_list) = args
    
    print('{0} -{1}- {2}'.format(time.asctime(time.localtime(time.time())),
                                 os.getpid(),
                                 multiprocessing.current_process().name))
    
    sv_dict = dict.fromkeys(sv_header)
        
    sv_dict['CHROM'] = rec.CHROM 
    sv_dict['POS'] = rec.POS 
    sv_dict['ID'] = rec.ID 
    sv_dict['REF'] = rec.REF 
    sv_dict['ALT'] = ','.join(map(lambda x: str(x),rec.ALT))
    sv_dict['QUAL'] = rec.QUAL 
    sv_dict['FILTER'] = 'PASS' if isinstance(rec.FILTER,list) else None
    
    for i_info in info_ids:
        if i_info in rec.INFO:
            sv_dict[i_info] = ','.join(map(lambda x: str(x),rec.INFO[i_info])) \
                                       if isinstance(rec.INFO[i_info],list) \
                                       else rec.INFO[i_info]
        else:
            sv_dict[i_info] = None
    
    sv_dict['FORMAT'] = rec.FORMAT 
    
    if sample_list:
        for i_sample in rec.samples:
            if i_sample.sample in sample_list:
            
                #print(i_sample.is_variant)
                sv_dict[i_sample.sample] = ','.join(map(lambda x: str(x),
                                                        [i_sample.gt_type, 
                                                         '|'.join(i_sample.gt_alleles) if list(
                                                             filter(None,i_sample.gt_alleles)) else '-1|-1',
                                                         int(i_sample.is_het) if i_sample.is_het else -1,
                                                         int(i_sample.is_variant) if i_sample.is_variant else -1]))
            else:
                sv_dict[i_sample.sample] = None

    sv_dict['SUPP_N_HET'] = rec.num_het
    sv_dict['SUPP_N_HOM_ALT'] = rec.num_hom_alt
    sv_dict['SUPP_IS_VAR'] = ','.join(
            [str(int(rec.is_deletion) if rec.is_deletion else -1),
             str(int(rec.is_indel) if rec.is_indel else -1),
             str(int(rec.is_snp) if rec.is_snp else -1),
             str(int(rec.is_sv) if rec.is_sv else -1)])
    sv_dict['SUPP_ALLES'] = ','.join(map(lambda x: str(x),rec.alleles if rec.alleles else None))
    
    sv_dict['SUPP_HETS'] = ';'.join([x.sample+','+'|'.join(x.gt_alleles) \
                                     if x else None for x in rec.get_hets()])
    sv_dict['SUPP_HOM_ALTS'] = ';'.join([x.sample+','+'|'.join(x.gt_alleles) \
                                         if x else None for x in rec.get_hom_alts()])
        
    return sv_dict


class VcfReader():
    
    def __init__(self, vcf_fname, chunk_size=100, dump_sample_gt=True):
        self.vcf_fname = vcf_fname
        self._v = vcf.Reader(filename=vcf_fname)
        
        # header for reading data
        # define header
        self._info_ids = list(self._v.infos.keys())
        fields1 = ['CHROM','POS','ID','REF','ALT','QUAL','FILTER']
        fields2 = ['FORMAT']
        fields3 = ['SUPP_N_HET','SUPP_N_HOM_ALT','SUPP_IS_VAR','SUPP_ALLES','SUPP_HETS','SUPP_HOM_ALTS']
        
        self.dump_sample_gt = dump_sample_gt
        if self.dump_sample_gt:
            self._sv_headers=np.concatenate([fields1,self._info_ids,fields2,self._v.samples,fields3])
        else:
            self._sv_headers=np.concatenate([fields1,self._info_ids,fields2,fields3])
        
        self._parse_data = parse_data
        
        self._chunk_size = chunk_size
        
    def get_metainfo(self):
        '''
        function to get meta-infos (start with ## in the file head) 
        from the vcf file
        '''
        
        meta_infos={}
        # alternations
        alts_df = pd.DataFrame(self._v.alts).T
        alts_df.columns=['ID','desc']
        alts_df.reset_index(drop=True, inplace=True)
        meta_infos['alts']=alts_df
        
        # Variant Level information
        var_infos_df = pd.DataFrame(self._v.infos).T
        var_infos_df.columns=['ID','num','type','desc','source','version']
        var_infos_df.reset_index(drop=True, inplace=True)
        meta_infos['var_info']=var_infos_df
        
        # Sample Level information such as GT,DP, etc.
        fmt_df = pd.DataFrame(self._v.formats).T
        fmt_df.columns=['ID','num','type','desc']
        fmt_df.reset_index(drop=True, inplace=True)
        meta_infos['meta_fmt']=fmt_df
        
        # chromosomes
        contigs_df = pd.DataFrame(self._v.contigs).T
        contigs_df.columns=['ID','length']
        contigs_df.reset_index(drop=True, inplace=True)
        meta_infos['contigs']=contigs_df
        
        # vcf meta
        vcfmeta_df = pd.DataFrame(self._v.metadata,index=[0])
        meta_infos['vcf_meta']=vcfmeta_df
        
        # filter
        filter_df = pd.DataFrame(self._v.filters).T
        filter_df.columns=['ID','desc']
        filter_df.reset_index(drop=True, inplace=True)
        meta_infos['meta_filter']=filter_df
        
        # file encoding
        meta_infos['encoding']=self._v.encoding
        
        # vcf file name
        meta_infos['filename']=self._v.filename
        
        # sample
        meta_infos['mate_samples']=self._v.samples
        
        return meta_infos
    
    def grouper(self, n, iterable, padvalue=None):
        """grouper(3, 'abcdefg', 'x') -->
        ('a','b','c'), ('d','e','f'), ('g','x','x')
        """
        return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)
    
    def run(self):
        '''
        function to read data from vcf file
        return pandas DataFrame
        '''
        
        # init a dict
        sv_dict = dict.fromkeys(self._sv_headers)
        for key in sv_dict.keys():
            sv_dict[key]=[] 
            
        p = multiprocessing.Pool(multiprocessing.cpu_count())
        
        # Use 'grouper' to split test data into
        # groups you can process without using a
        # ton of RAM. You'll probably want to 
        # increase the chunk size considerably
        # to something like 1000 lines per core.
        # The idea is that you replace 'test_data'
        # with a file-handle
        # e.g., testdata = open(file.txt,'rU')
        
        # And, you'd write to a file instead of
        # printing to the stout
        
        
        for i,chunk in enumerate(self.grouper(self._chunk_size, self._v)):
            
            print('Processing {0}th chunk data (chunk_size={1})...'.format(i, self._chunk_size))
            
            if self.dump_sample_gt:
                args = (self._sv_headers,self._info_ids,self._v.samples)
            else:
                args = (self._sv_headers,self._info_ids,None)
                
            chunk_results = p.starmap(self._parse_data, zip(chunk, repeat(args)))

            
            for re in chunk_results:
                if re is None:
                    continue
                sv_dict['CHROM'].append(re['CHROM'])
                sv_dict['POS'].append(re['POS'])
                sv_dict['ID'].append(re['ID'])
                sv_dict['REF'].append(re['REF'])
                sv_dict['ALT'].append(re['ALT'])
                sv_dict['QUAL'].append(re['QUAL'])
                sv_dict['FILTER'].append(re['FILTER'])
                
                for i_info in self._info_ids:
                    sv_dict[i_info].append(re[i_info])
                
                sv_dict['FORMAT'].append(re['FORMAT'])
                
                if self.dump_sample_gt:
                    for i_sample in self._v.samples:
                        sv_dict[i_sample].append(re[i_sample])
                
                sv_dict['SUPP_N_HET'].append(re['SUPP_N_HET'])
                sv_dict['SUPP_N_HOM_ALT'].append(re['SUPP_N_HOM_ALT'])
                sv_dict['SUPP_IS_VAR'].append(re['SUPP_IS_VAR'])
                
                sv_dict['SUPP_ALLES'].append(re['SUPP_ALLES'])
                sv_dict['SUPP_HETS'].append(re['SUPP_HETS'])
                sv_dict['SUPP_HOM_ALTS'].append(re['SUPP_HOM_ALTS'])
        
        p.close()
        p.join()
        
        sv_df = pd.DataFrame(sv_dict)
        # calculate regeion length
        sv_df['tmpLEN']=sv_df['END']-sv_df['POS']
        sv_df.loc[sv_df['SVLEN'].isnull(),'SVLEN'] = sv_df.loc[sv_df['SVLEN'].isnull(),'tmpLEN']
        sv_df.drop(['tmpLEN'],axis=1,inplace=True)
        
        return sv_df
    
