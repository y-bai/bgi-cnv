#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: load_data.py
    Description:
    
Created by Yong Bai on 2019/9/18 10:55 PM.
"""

import numpy as np
import keras


class CNVDataGenerator(keras.utils.Sequence):

    def __init__(self, data_sample_df, batch_size, win_size=1000, n_feat=13, n_classes=3, shuffle=True, pred_gen=False):

        self.data_sample_df = data_sample_df
        self.data_sample_len = len(self.data_sample_df)
        # self.data_sample_idx = self.data_sample_df.index.values
        self.batch_size = batch_size
        self.win_size = win_size
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.pred_gen = pred_gen

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

        batch_sample_df = self.data_sample_df.iloc[sel_idx]

        return self.__data_generation(batch_sample_df)

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        :return:
        """
        self.data_sample_idx = self.data_sample_df.index.values
        if self.shuffle:
            np.random.shuffle(self.data_sample_idx)

    def __data_generation(self, batch_sample_df):
        """
        Generates data containing batch_size samples
        :param batch_sample_df:
        :return:
        """
        # Initialization
        x = np.empty((self.batch_size, self.win_size, self.n_feat), dtype=np.float32)
        if not self.pred_gen:
            y = np.empty(self.batch_size, dtype=np.int)

        # Generate data
        for i, (row_ind, i_row) in enumerate(batch_sample_df.iterrows()):
            # Store sample
            with np.load(i_row['f_name']) as i_feat_map:
                x[i, ] = i_feat_map['x']

            # Store class
            if not self.pred_gen:
                y[i] = i_row['cnv_type_encode']

        if not self.pred_gen:
            return x, keras.utils.to_categorical(y, num_classes=self.n_classes)
        else:
            return x
