#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: network.py
    Description:
    
Created by Yong Bai on 2019/8/20 8:30 PM.
"""

import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Flatten, Dropout, MaxPooling1D, Activation, BatchNormalization
from keras.layers import Add, Multiply, GlobalAveragePooling1D, Reshape


def _bn_relu(x):
    """
    Helper to build a BN -> relu block
    :param x:
    :return:
    """
    norm = BatchNormalization()(x)
    return Activation("relu")(norm)


def basic_residual_unit(**kwargs):
    """
    Helper to build a BN->relu->conv1d residual unit
    :param res_params:
    :return:
    """

    filters = kwargs['filters']
    kernel_size = kwargs.setdefault('kernel_size', 16)
    strides = kwargs.setdefault('strides', 1)

    def f(x):
        x = Conv1D(filters=filters,
                   kernel_size=kernel_size,
                   padding='same',
                   strides=strides,
                   kernel_initializer='he_normal')(x)

        return _bn_relu(x)

    return f


def preact_residual_unit(**kwargs):
    """

    :param kwargs:
    :return:
    """
    filters = kwargs['filters']
    kernel_size = kwargs.setdefault('kernel_size', 16)
    strides = kwargs.setdefault('strides', 1)
    drop = kwargs.setdefault('drop', 0.5)

    def f(x):
        x = _bn_relu(x)
        x = Dropout(drop)(x)  # the opt place where to put drop
        return Conv1D(filters=filters,
                      kernel_size=kernel_size,
                      padding='same',
                      strides=strides,
                      kernel_initializer='he_normal')(x)  # regularizer will degrade performance

    return f


def residual_block(input_x, idx_block, n_block, **kwargs):
    """

    :param input_x:
    :param idx_block:
    :param n_block:
    :param kwargs:
    :return:
    """
    init_filters = kwargs['filters']
    filters = init_filters * (2 ** idx_block)
    kwargs['filters'] = filters

    x = input_x
    x_shortcut = Conv1D(filters=filters, kernel_size=1)(x) if idx_block != 0 else x

    for i in range(n_block):
        x1 = preact_residual_unit(**kwargs)(x)
        # x1 = Dropout(drop)(x1)  # if do drop here, then train converge too fast, and lead to overfitting
        x1 = preact_residual_unit(**kwargs)(x1)

        # must do pool like this, otherwise local opt.
        if (i + 1) % 2 == 0:
            x1 = MaxPooling1D(pool_size=kwargs['pool_size'], strides=kwargs['pool_stride'])(x1)
            x2 = MaxPooling1D(pool_size=kwargs['pool_size'], strides=kwargs['pool_stride'])(x_shortcut)
        else:
            x2 = x_shortcut

        x1 = cse_sse_block(x1)
        x = keras.layers.add([x1, x2])
        x_shortcut = x

    return x


def se_block(se_input, ratio=16):
    """
    Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. CVPR(2018)

    Implementation adapted from
        https://github.com/titu1994/keras-squeeze-excite-network/blob/master/se.py
    :param se_input:
    :param ratio:
    :return:
    """

    init = se_input
    filters = init._keras_shape[-1]
    se_shape = (1, filters)

    se = GlobalAveragePooling1D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)

    x = Multiply()([init, se])
    return x


def spatial_se_block(ses_input):
    """
    Create a spatial squeeze-excite block
    Ref:
        [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks]
    (https://arxiv.org/abs/1803.02579)
    :param ses_input:
    :return:
    """

    se = Conv1D(1, 1, activation='sigmoid', use_bias=False)(ses_input)
    x = Multiply()([ses_input, se])
    return x


def cse_sse_block(se_input, ratio=16):
    """
    Create a spatial squeeze-excite block
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks]
        (https://arxiv.org/abs/1803.02579)
    :param se_input:
    :param ratio:
    :return:
    """

    cse = se_block(se_input, ratio)
    sse = spatial_se_block(se_input)

    x = Add()([cse, sse])
    return x


def simple_residual_block(input_x, **kwargs):
    filters = kwargs['filters']
    kernel_size = kwargs.setdefault('kernel_size', 16)
    strides = kwargs.setdefault('strides', 1)
    drop = kwargs.setdefault('drop', 0.5)
    pool_size = kwargs['pool_size']
    pool_stride = kwargs['pool_stride']

    # left branch
    x1 = Conv1D(filters=filters,
                kernel_size=kernel_size,
                padding='same',
                strides=strides,
                kernel_initializer='he_normal')(input_x)
    x1 = Dropout(drop)(x1)
    x1 = _bn_relu(x1)
    x1 = Conv1D(filters=filters,
                kernel_size=kernel_size,
                padding='same',
                strides=strides,
                kernel_initializer='he_normal')(x1)
    x1 = Dropout(drop)(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(x1)

    # right branch
    x2 = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(input_x)

    x1 = cse_sse_block(x1)
    x = keras.layers.add([x1, x2])
    del x1, x2
    return x


def cnv_net(win_size, n_in_feat, n_out_class, filters=64, kernel_size=16, strides=1, pool_size=2,
            pool_stride=2, drop=0.2, blocks=(4, 1)):  #
    """

    :param win_size:
    :param n_in_feat:
    :param n_out_class:
    :param filters:
    :param kernel_size:
    :param strides:
    :param pool_size:
    :param pool_stride:
    :param drop:
    :param blocks:
    :return:
    """
    input_x = Input(shape=(win_size, n_in_feat), name='input')
    x = basic_residual_unit(filters=filters, kernel_size=kernel_size, strides=strides)(input_x)

    for j, n_block in enumerate(blocks):
        x = residual_block(x, j, n_block, filters=filters, kernel_size=kernel_size, strides=strides,
                           pool_size=pool_size, pool_stride=pool_stride, drop=drop)
    x = _bn_relu(x)
    # x = GlobalAveragePooling1D()(x)  #lead to overfitting
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    out = Dense(n_out_class, activation='softmax')(x)
    model = Model(inputs=input_x, outputs=out)
    return model


def cnv_simple_net(win_size, n_in_feat, n_out_class, filters=64, kernel_size=16, strides=1, pool_size=2,
                   pool_stride=2, drop=0.5):  #
    """
        this model will easily be overfitting or hard to training
    :param win_size:
    :param n_in_feat:
    :param n_out_class:
    :param filters:
    :param kernel_size:
    :param strides:
    :param pool_size:
    :param pool_stride:
    :param drop:
    :return:
    """

    input_x = Input(shape=(win_size, n_in_feat), name='input')
    x = basic_residual_unit(filters=filters, kernel_size=kernel_size, strides=strides)(input_x)

    for i in range(5):
        x = simple_residual_block(x, filters=filters, kernel_size=kernel_size, strides=strides,
                                  pool_size=pool_size, pool_stride=pool_stride, drop=drop)
        # x = cse_sse_block(x)
    x = basic_residual_unit(filters=filters, kernel_size=kernel_size, strides=strides)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu', )(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(n_out_class, activation='softmax')(x)
    model = Model(inputs=input_x, outputs=out)
    return model
