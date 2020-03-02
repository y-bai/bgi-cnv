#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: train_test_data.py
    Description:
    
Created by Yong Bai on 2019/8/20 3:04 PM.
"""
import os
import gc
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from keras.utils import to_categorical, Sequence

import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_data(feature_file, train_id):
    """
    load cnv feature from npz file
    """
    
    del_dup_data_fn = os.path.join(os.path.split(feature_file[0])[0], 
                "ds_data4model_{0}".format(train_id) + str(os.path.split(feature_file[0])[1]))
    
    neu_data_fn = os.path.join(os.path.split(feature_file[1])[0], 
                "ds_data4model_{0}".format(train_id) + str(os.path.split(feature_file[1])[1]))

    # logger.info('Loading dataset for sample {}...'.format(train_id))
    if os.path.exists(del_dup_data_fn) and os.path.exists(neu_data_fn):
        with np.load(del_dup_data_fn) as del_dup_data, np.load(neu_data_fn) as neu_data:
            x = np.concatenate((del_dup_data['x'], neu_data['x']), axis=0)
            y = np.concatenate((del_dup_data['y'], neu_data['y']))

        return x, y

    else:
        logger.warning('sample {} does not have enough data.'.format(train_id))
        return None, None


def get_train_test_ids(model_root_dir, total_samples_ls_fn, test_size=0.3):
    """

    :param model_root_dir:
    :param total_samples_ls_fn:
    :param test_size:
    :return:
    """

    # create the train and test sample ids
    logger.info("Loading training and testing sample ids...")
    train_test_sample_list = os.path.join(model_root_dir, '1k_train_test{:.2f}_sample_ids.npz'.format(test_size))
    if not os.path.exists(train_test_sample_list):
        if not os.path.exists(total_samples_ls_fn):
            raise FileNotFoundError('sample list file does not exist. {}'.format(total_samples_ls_fn))
        else:
            sample_id_map = np.loadtxt(total_samples_ls_fn, delimiter='\t', usecols=(0, 1), dtype='str')
            sample_id_arr = sample_id_map[:, 0]
            train_ids, test_ids = train_test_split(sample_id_arr, test_size=test_size, random_state=123)
            np.savez(train_test_sample_list, train_ids=train_ids, test_ids=test_ids)
            logger.info("Total training sample {}...".format(len(train_ids)))
            logger.info("Total testing sample {}...".format(len(test_ids)))
            return train_ids, test_ids
    else:
        with np.load(train_test_sample_list) as sample_ids:
            logger.info("Total training sample {}...".format(len(sample_ids['train_ids'])))
            logger.info("Total testing sample {}...".format(len(sample_ids['test_ids'])))
            return sample_ids['train_ids'], sample_ids['test_ids']


def load_train_data(model_root_dir, total_samples_ls_fn, win_size,
                    min_r, min_f_deldup, min_f_neu, is_norm, test_ratio=0.3, min_size=100000):
    """

    :param model_root_dir:
    :param total_samples_ls_fn:
    :param win_size:
    :param min_r:
    :param min_f_deldup:
    :param min_f_neu:
    :param is_norm:
    :param min_sze:
    :return:
    """

    train_set_fn = os.path.join(model_root_dir,
                                'train{:.2f}_win{}_minsize{}_dataset.npz'.format(1-test_ratio, win_size, min_size))
    if os.path.exists(train_set_fn):
        with np.load(train_set_fn) as train_set:
            return train_set['x_train'], train_set['y_train']

    train_ids, _ = get_train_test_ids(model_root_dir, total_samples_ls_fn, test_ratio)
    x_train = []
    y_train_ = []
    total_len=[]
    for ix, train_id in enumerate(train_ids):
        x, y = get_data(train_id, model_root_dir, win_size, min_r, min_f_deldup, min_f_neu, is_norm)
        total_len = len(y) + total_len
        if (not (x is None)) and (not (y is None)):

            # deal with data balance
            del_idx = np.where(y == 'DEL')[0]
            dup_idx = np.where(y == 'DUP')[0]
            neu_idx = np.where(y == 'NEU')[0]

            len_del_idx = len(del_idx)
            len_dup_idx = len(dup_idx)
            len_neu_idx = len(neu_idx)

            if len_del_idx == 0 or len_dup_idx == 0 or len_neu_idx == 0:
                logger.warning('del len: {}, dup len: {}, neu len: {}'.format(len_del_idx, len_dup_idx, len_neu_idx))
                continue

            logger.info('del len: {}, dup len: {}, neu len: {}'.format(len_del_idx, len_dup_idx, len_neu_idx))
            min_idx_len = np.min([len(del_idx), len(dup_idx), len(neu_idx)])

            f_del_idx = np.random.choice(del_idx, min_idx_len, replace=False)
            f_dup_idx = np.random.choice(dup_idx, min_idx_len, replace=False)
            f_neu_idx = np.random.choice(neu_idx, min_idx_len, replace=False)

            f_idx = np.concatenate((f_del_idx, f_dup_idx, f_neu_idx))
            y_ = y[f_idx]
            x_ = x[f_idx, :, :]

            if len(x_train) == 0:
                x_train = x_
                y_train_ = y_
            else:
                x_train = np.concatenate((x_train, x_), axis=0)
                y_train_ = np.concatenate((y_train_, y_))

            if len(y_train_) >= 3*min_size:
                break
    y = [1 if x == 'DEL' else 2 if x == 'DUP' else 0 for x in y_train_]
    y_train = to_categorical(y)
    del y_train_
    gc.collect()
    logger.info('x train: {}, y train: {}'.format(x_train.shape, y_train.shape))
    np.savez_compressed(train_set_fn, x_train=x_train, y_train=y_train)
    return x_train, y_train


x_temp = None
y_temp =None 
last_sample = ''

def my_generator(model_root_dir, total_samples_ls_fn, win_size,
                    min_r, min_f_deldup, min_f_neu, is_norm, test_ratio=0.3, validation_ratio=0.2, batch_size=36, n_classes = 3, shuffle=None):
    

    
    train_ids, _ = get_train_test_ids(model_root_dir, total_samples_ls_fn, test_ratio)
    logger.info('The trainning and validation contains a total of {} human samples ...'.format(len(train_ids)))
    train_validation_cnv_sample_list = os.path.join(model_root_dir, '1k_train_validation{:.2f}_cnv_sample_ids.json'.format(validation_ratio))
    
    #data dir set
    data_dir = os.path.join(model_root_dir, 'data4models')
    del_dup_data_fn = os.path.join(data_dir, "_{0}_{1:.2f}_{2:.2f}_{3}.npz".format(
        win_size, min_r, min_f_deldup, is_norm))
    neu_data_fn = os.path.join(data_dir, "_{0}_{1:.2f}_{2:.2f}_{3}.npz".format(
        win_size, min_r, min_f_neu, is_norm))
    
    feature_file = [del_dup_data_fn, neu_data_fn]
    #extract cnv feature from npz file and generate cnv ids with sample id in all train and validation human samples
    if not os.path.exists(train_validation_cnv_sample_list):
        cnv_samples_x = []
        cnv_samples_y = []
        logger.info('First run trainning step on validation rate {:.2f} , now begin loading trainning data...'.format(validation_ratio))
        for ix, train_id in enumerate(train_ids):

            logger.info('Loading dataset for sample {}...'.format(train_id))
            #data path

            x, y = get_data(feature_file, train_id)
            #combine cnvs in every human samples
            indexes = list(range(x.shape[0])) 
            #shuffle CNVS in sample
            if shuffle == 'Local_shuffle':
                np.random.shuffle(indexes)
            cnv_samples_x = cnv_samples_x + [str(train_id) + '.__.' + str(x) for x in indexes]
            cnv_samples_y = cnv_samples_y + [y[x] for x in indexes]

        X_train_cnv_samples, X_test_cnv_samples, Y_train_cnv_samples, Y_test_cnv_samples = train_test_split(cnv_samples_x, 
                                                                                                            cnv_samples_y, 
                                                                                                            test_size=validation_ratio,
                                                                                                            random_state=123)
        
        #Save Dataset to json file                                                                          
        Dataset = {'train cnv samples':[X_train_cnv_samples, Y_train_cnv_samples],
                    'validation cnv samples':[X_test_cnv_samples,Y_test_cnv_samples]}
        with open(train_validation_cnv_sample_list, 'w') as f:
            json.dump(Dataset, f)

    else:
        #load Dataset
        logger.info('Loading data from exist file on validation rate {:.2f} ...'.format(validation_ratio))
        with open(train_validation_cnv_sample_list) as f:
            Dataset = json.load(f)
    
    
    #calculate 3 kind CNV types for trainning and validation
    my_gen=[]
    for dataset in Dataset.keys():
        del_idx = Dataset[dataset][1].count('DEL')
        dup_idx = Dataset[dataset][1].count('DUP')
        neu_idx = Dataset[dataset][1].count('NEU')

        logger.info('The {} set contains a total of {} DUP samples ...'.format(dataset,dup_idx))
        logger.info('The {} set contains a total of {} DEL samples ...'.format(dataset,del_idx))
        logger.info('The {} set contains a total of {} NEU samples ...'.format(dataset,neu_idx))

        params = {'feature_path':feature_file,
            'dim': win_size,
            'batch_size': batch_size,
            'n_classes': n_classes,
            'n_channels': 13,
            'shuffle': shuffle}

        logger.info("Generating training and validation features...")
        train_generator = DataGenerator(Dataset[dataset][0],**params)
        my_gen.append(train_generator)
    return my_gen[0],my_gen[1]


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, sample_IDs, feature_path, batch_size=3, dim=1000, n_channels=13,
                 n_classes=3, shuffle=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.feature_path = feature_path
        self.list_IDs = sample_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] 
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        
        #shuffle cnvs among all trainning human samples
        if self.shuffle == 'Global_shuffle':
            np.random.shuffle(self.indexes)
        

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim, self.n_channels))
        Y = np.empty((self.batch_size), dtype=int)

        # Generate data

        global x_temp, y_temp, last_sample
        label2nums={'DEL':1,'DUP':2,'NEU':0}
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            sample_id = ID.split('.__.', 1)
            print(sample_id[0])
            if last_sample != sample_id[0]:
                print("------------import data from file")
                x_temp, y_temp = get_data(self.feature_path, sample_id[0])
            print("############one cnv samples###########")   
            last_sample = sample_id[1]
            
            X[i,] = x_temp[int(sample_id[1]), :, :]
            Y[i]=label2nums[y_temp[int(sample_id[1])]]

        Y = to_categorical(Y, num_classes=self.n_classes)

        return X, Y
        