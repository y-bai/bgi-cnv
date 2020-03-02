#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: resnet_archi.py
    Description:
    This model's architecture is exactly the same to the ResNet.
    Unfortunately, the performance is too bad.
Created by Yong Bai on 2019/8/15 3:39 PM.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, BatchNormalization, Activation, GlobalAveragePooling1D, Reshape
from keras.layers import Conv1D, Dropout, MaxPooling1D, ZeroPadding1D, Flatten, Dense, Add, Multiply
from keras.regularizers import l2
from keras.models import Model


def _bn_relu(x, bn_name=None, relu_name=None):
    """
    Helper to build a BN -> relu block
    :param x:
    :param bn_name:
    :param relu_name:
    :return:
    """
    norm = BatchNormalization(name=bn_name)(x)
    return Activation("relu", name=relu_name)(norm)


def se_block(se_input, ratio=16, name=None):
    """
    Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. CVPR(2018)

    Implementation adapted from
        https://github.com/titu1994/keras-squeeze-excite-network/blob/master/se.py
    :param se_input:
    :param ratio:
    :param name
    :return:
    """
    init = se_input
    filters = init._keras_shape[-1]
    se_shape = (1, filters)

    se = GlobalAveragePooling1D(name=name + '_gvp')(init)
    se = Reshape(se_shape, name=name + '_reshape')(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False,
               name=name + '_relu_dense')(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False,
               name=name + '_sig_dense')(se)

    x = Multiply(name=name + '_mul')([init, se])
    return x


def spatial_se_block(ses_input, name=None):
    """
    Create a spatial squeeze-excite block
    Ref:
        [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks]
    (https://arxiv.org/abs/1803.02579)
    :param ses_input:
    :param name:
    :return:
    """

    se = Conv1D(1, 1, activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal', name=name+'_conv')(ses_input)

    x = Multiply(name=name+'_mul')([ses_input, se])
    return x


def cse_sse_block(se_input, ratio=16, name=None):
    """
    Create a spatial squeeze-excite block
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks]
        (https://arxiv.org/abs/1803.02579)
    :param input:
    :param ratio:
    :param name:
    :return:
    """

    cse = se_block(se_input, ratio, name=name+'_cse')
    sse = spatial_se_block(se_input, name=name+'_sse')

    x = Add(name=name + '_se_add')([cse, sse])
    return x


def bottleneck(x, filters, kernel_size=7, stride=1, conv_shortcut=False, name=None, dropout=None, attention=False):
    """
    A residual block with or without attention gate

    The network architecture follow
        [Identity Mappings in Deep Residual Networks](http://arxiv.org/pdf/1603.05027v2.pdf)

    Implementation adapted from
        https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py

    :param x:
        input tensor
    :param filters:
        integer, filters of the bottleneck layer.
    :param kernal_size:
        default 16, kernel size of the bottleneck layer.
    :param stride:
        default 1, stride of the first layer.
    :param conv_shortcut:
        default False, use convolution shortcut if True, otherwise identity shortcut.
    :param name:
        string, block label.
    :param dropout:
        float, dropout ratio.
    :param attention:
        default False, use senet+resent for attention gate if True, otherwise resnet.
    :return:
        Output tensor for the residual block.
    """

    # pre-active unit: 1 x 1 conv, 248
    preact = _bn_relu(x, bn_name=name + '_preact_bn', relu_name=name + '_preact_relu')
    if conv_shortcut is True:
        shortcut = Conv1D(filters=4 * filters, kernel_size=1, strides=stride,
                          name=name + '_0_conv')(preact)
    else:
        shortcut = MaxPooling1D(pool_size=1, strides=stride)(x) if stride > 1 else x

    x = Conv1D(filters=filters, kernel_size=1, strides=1, use_bias=False,
               name=name + '_1_conv')(preact)
    # The reason for placing Dropout here is to following
    # https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/resnet.py
    if dropout is not None:
        x = Dropout(dropout, name=name + '_1_dropout')(x)

    # second unit: conv
    x = _bn_relu(x, bn_name=name + '_2_bn', relu_name=name + '_2_relu')

    x = ZeroPadding1D(padding=3, name=name + '_2_pad')(x)
    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, use_bias=False,
               kernel_initializer='he_normal',
               name=name + '_2_conv')(x)
    if dropout is not None:
        x = Dropout(dropout, name=name + '_2_dropout')(x)

    # third unit: 1 x 1 conv
    x = _bn_relu(x, bn_name=name + '_3_bn', relu_name=name + '_3_relu')

    x = Conv1D(filters=4 * filters, kernel_size=1, strides=1,
               name=name + '_3_conv')(x)

    if attention:
        x = cse_sse_block(x, name=name+'_att')

    x = Add(name=name + '_out')([shortcut, x])

    return x


def resnet_block(x, filters, nb_blocks, nb_stride=2, name=None, dropout=None, attention=False):
    """
    A set of stacked residual blocks.
    :param x:
        input tensor
    :param filters:
        integer, filters of the bottleneck layer in a block.
    :param nb_blocks:
        integer, blocks in the stacked blocks.
    :param nb_stride:
        default 2, stride of the first layer in the first block.
    :param name:
        string, stack label.
    :param dropout:
    :return:
        Output tensor for the stacked blocks.
    """

    x = bottleneck(x, filters, conv_shortcut=True, name=name + '_block1', dropout=dropout, attention=attention)
    for i in range(2, nb_blocks):
        x = bottleneck(x, filters, name=name + '_block' + str(i), dropout=dropout, attention=attention)
    x = bottleneck(x, filters, stride=nb_stride, name=name + '_block' + str(nb_blocks),
                   dropout=dropout, attention=attention)

    return x


def ResNet1D(win_size, n_in_feat, n_out_class, model_name='cnv_resnet1d',
             init_filters=32, init_kernel_size=17, dropout=0.5,
             attention=False):
    """

    :param input_shape:
    :param n_class:
    :param model_name:
    :param init_filters:
    :param init_kernel_size:
    :param dropout:
    :param attention:
    :return:
    """
    input_shape = (win_size, n_in_feat)
    input_x = Input(shape=input_shape, name='input')

    # first conv block
    x = ZeroPadding1D(padding=(4, 4), name='conv1_pad')(input_x)
    x = Conv1D(filters=init_filters, kernel_size=init_kernel_size, strides=2,
               kernel_initializer='he_normal', name='conv1_conv')(x)
    # now, the feature map size = (496, 32) is input_shape[0]=1000
    x = _bn_relu(x, bn_name='conv1_bn', relu_name='conv1_relu')

    x = ZeroPadding1D(padding=1, name='pool1_pad')(x)
    x = MaxPooling1D(3, strides=2, name='pool1_pool')(x)
    # now the feature map size = (248, 32)

    # resnet stacks
    x = resnet_block(x, init_filters * 1, 3, name='conv2', dropout=dropout, attention=attention)
    x = resnet_block(x, init_filters * 2, 4, name='conv3', dropout=dropout, attention=attention)
    x = resnet_block(x, init_filters * 4, 6, name='conv4', dropout=dropout, attention=attention)
    x = resnet_block(x, init_filters * 8, 3, name='conv5', dropout=dropout, attention=attention)

    x = _bn_relu(x, bn_name='post_bn', relu_name='post_relu')

    x = Flatten(name='final_out_flatten')(x)
    x = Dense(1000, activation='relu', name='final_out_dense1')(x)
    x = Dense(1000, activation='relu', name='final_out_dense2')(x)
    # x = GlobalAveragePooling1D(name='final_out_avg')(x)
    out = Dense(n_out_class, activation='softmax', name='final_out_softmax')(x)

    model = Model(inputs=input_x, outputs=out, name=model_name)

    return model
