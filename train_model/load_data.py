#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: load_data.py
    Description:
    
Created by Yong Bai on 2019/9/18 10:55 PM.
"""

import numpy as np
import keras
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py


class CNVDataGenerator(keras.utils.Sequence):

    def __init__(self, data_sample_arr, batch_size,
                 win_size=1000, n_feat=13, n_classes=3, shuffle=True, pred_gen=False, smooth_factor=0.1):

        self.data_sample_arr = data_sample_arr
        self.data_sample_len = len(self.data_sample_arr)
        # self.data_sample_idx = self.data_sample_df.index.values
        self.batch_size = batch_size
        self.win_size = win_size
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.pred_gen = pred_gen
        self.smooth_factor = smooth_factor

        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return:
        """
        if not self.pred_gen:
            return int(np.floor(self.data_sample_len / float(self.batch_size)))
        else:
            return int(np.ceil(self.data_sample_len / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Generate one batch of data
        :param idx:
        :return:
        """

        sel_idx = self.data_sample_idx[idx * self.batch_size:(idx + 1) * self.batch_size]

        i_batch_sample_arr = self.data_sample_arr[sel_idx]

        # gc.collect()
        return self.__data_generation(i_batch_sample_arr)

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        :return:
        """
        self.data_sample_idx = np.arange(self.data_sample_len)
        if self.shuffle:
            np.random.shuffle(self.data_sample_idx)

    def __data_generation(self, i_batch_sample_arr):
        """
        Generates data containing batch_size samples
        :param i_batch_sample_arr:
        :return:
        """
        # Initialization
        x = np.empty((self.batch_size, self.win_size, self.n_feat), dtype=np.float32)
        if not self.pred_gen:
            y = np.empty(self.batch_size, dtype=np.int)

        # Generate data
        for i, i_row in enumerate(i_batch_sample_arr):
            # Store sample
            with np.load(i_row[0]) as i_feat_map:
                x[i, ] = i_feat_map['x'].astype(np.float32)

            # Store class
            if not self.pred_gen:
                y[i] = int(i_row[1])

        if not self.pred_gen:
            y_mat = keras.utils.to_categorical(y, num_classes=self.n_classes)
            if 0 <= self.smooth_factor <= 1:
                y_mat *= 1 - self.smooth_factor
                y_mat += self.smooth_factor / y_mat.shape[1]
            return x, y_mat
        else:
            return x


class CNVDataGenerator2(keras.utils.Sequence):

    def __init__(self, data_h5_fn, data_sample_len, batch_size,
                 win_size=1000, n_feat=13, n_classes=3, shuffle=True, pred_gen=False, smooth_factor=0.1):

        self.data_h5_fn = data_h5_fn

        self.data_sample_len = data_sample_len
        # self.data_sample_idx = self.data_sample_df.index.values
        self.batch_size = batch_size
        self.win_size = win_size
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.pred_gen = pred_gen
        self.smooth_factor = smooth_factor

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return:
        """
        if not self.pred_gen:
            return int(np.floor(self.data_sample_len / float(self.batch_size)))
        else:
            return int(np.ceil(self.data_sample_len / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Generate one batch of data
        :param idx:
        :return:
        """
        return self.__data_generation(idx * self.batch_size, (idx + 1) * self.batch_size)

    def __data_generation(self, batch_start, batch_end):
        """
        Generates data containing batch_size samples
        :param i_batch_sample_arr:
        :return:
        """
        # Initialization
        x = np.empty((self.batch_size, self.win_size, self.n_feat), dtype=np.float32)
        if not self.pred_gen:
            y = np.empty(self.batch_size, dtype=np.int)
        # Generate data
        # def f(x):
        #     cnv_tpye = x.split('|')[5]
        #     return 1 if cnv_tpye == 'DEL' else 2 if cnv_tpye == 'DUP' else 0

        with h5py.File(self.data_h5_fn, 'r') as data_h5:
            x[...] = data_h5['x'][batch_start:batch_end, ...]
            y[...] = data_h5['y'][batch_start:batch_end, ...]

        if not self.pred_gen:
            y_mat = keras.utils.to_categorical(y, num_classes=self.n_classes)
            if 0 <= self.smooth_factor <= 1:
                y_mat *= 1 - self.smooth_factor
                y_mat += self.smooth_factor / y_mat.shape[1]
            return x, y_mat
        else:
            return x


class CNVDataGenerator3(keras.utils.Sequence):

    def __init__(self,  data_h5_fn, data_sample_len, batch_size,
                 win_size=1000, n_feat=13, n_classes=3, shuffle=True, pred_gen=False, smooth_factor=0.1):

        self.data_h5_fn = data_h5_fn

        self.data_sample_len = data_sample_len
        # self.data_sample_idx = self.data_sample_df.index.values
        self.batch_size = batch_size
        self.win_size = win_size
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.pred_gen = pred_gen
        self.smooth_factor = smooth_factor

        # self.lock = threading.Lock()

        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return:
        """
        if not self.pred_gen:
            return int(np.floor(self.data_sample_len / float(self.batch_size)))
        else:
            return int(np.ceil(self.data_sample_len / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Generate one batch of data
        :param idx:
        :return:
        """
        # i_batch_sample_arr = self.data_sample_arr[sel_idx]

        # avoid possible deadlock when reach epoch end
        # ref: https://www.kaggle.com/ezietsman/keras-convnet-with-fit-generator
        # with self.lock:
        sel_idx = self.data_sample_idx[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.__data_generation(sel_idx)

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        :return:
        """
        self.data_sample_idx = np.arange(self.data_sample_len)

        if self.shuffle:
            np.random.shuffle(self.data_sample_idx)

    def __data_generation(self, i_batch_sample_indx):
        """
        Generates data containing batch_size samples
        :param i_batch_sample_arr:
        :return:
        """
        # Initialization
        x = np.empty((self.batch_size, self.win_size, self.n_feat), dtype=np.float32)
        if not self.pred_gen:
            y = np.empty(self.batch_size, dtype=np.int)

        # Generate data
        sort_idx = np.argsort(i_batch_sample_indx)
        back_idx = np.argsort(sort_idx)

        after_sort = i_batch_sample_indx[sort_idx]
        sorted_idx = list(after_sort)
        # would lead to process hang without move forward
        with h5py.File(self.data_h5_fn, 'r') as data_h5:
            x[...] = data_h5['x'][sorted_idx, ...]
            y[...] = data_h5['y'][sorted_idx, ...]
        x = x[back_idx]
        y = y[back_idx]

        if not self.pred_gen:
            y_mat = keras.utils.to_categorical(y, num_classes=self.n_classes)
            if 0 <= self.smooth_factor <= 1:
                y_mat *= 1 - self.smooth_factor
                y_mat += self.smooth_factor / y_mat.shape[1]
            return x, y_mat
        else:
            return x

class CNVDataGenerator_AE(keras.utils.Sequence):
    """
    for Autoecoder
    """
    def __init__(self,  data_h5_fn, data_sample_len, batch_size,
                 win_size=1000, n_feat=9, shuffle=True, pred_gen=False, smooth_factor=0.1):

        self.data_h5_fn = data_h5_fn
        self.data_sample_len = data_sample_len
        # self.data_sample_idx = self.data_sample_df.index.values
        self.batch_size = batch_size
        self.win_size = win_size
        self.n_feat = n_feat
        self.shuffle = shuffle
        self.pred_gen = pred_gen
        self.smooth_factor = smooth_factor
        # self.lock = threading.Lock()

        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return:
        """
        if not self.pred_gen:
            return int(np.floor(self.data_sample_len / float(self.batch_size)))
        else:
            return int(np.ceil(self.data_sample_len / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Generate one batch of data
        :param idx:
        :return:
        """

        # i_batch_sample_arr = self.data_sample_arr[sel_idx]

        # avoid possible deadlock when reach epoch end
        # ref: https://www.kaggle.com/ezietsman/keras-convnet-with-fit-generator
        # with self.lock:
        end_slice = min((idx + 1) * self.batch_size, self.data_sample_len)
        sel_idx = self.data_sample_idx[idx * self.batch_size:end_slice]
        return self.__data_generation(sel_idx)

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        :return:
        """
        self.data_sample_idx = np.arange(self.data_sample_len)

        if self.shuffle:
            np.random.shuffle(self.data_sample_idx)

    def __data_generation(self, i_batch_sample_indx):
        """
        Generates data containing batch_size samples
        :param i_batch_sample_arr:
        :return:
        """
        # Initialization
        # x = np.empty((self.batch_size, self.win_size, self.n_feat), dtype=np.float32)
        n_sample_ibatch = len(i_batch_sample_indx)
        x = np.empty((n_sample_ibatch, self.win_size, self.n_feat), dtype=np.float32)

        # Generate data
        sort_idx = np.argsort(i_batch_sample_indx)
        back_idx = np.argsort(sort_idx)

        after_sort = i_batch_sample_indx[sort_idx]
        sorted_idx = list(after_sort)
        # would lead to process hang without move forward
        with h5py.File(self.data_h5_fn, 'r') as data_h5:
            x[...] = data_h5['x'][sorted_idx, :, 0:self.n_feat]
        x = x[back_idx]

        if not self.pred_gen:
            return x, x
        else:
            return x

