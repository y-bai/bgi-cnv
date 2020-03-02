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
from keras.layers import Add, Multiply, GlobalAveragePooling1D, Reshape, AveragePooling1D, UpSampling1D
from keras.layers import GRU, LSTM, CuDNNLSTM, Bidirectional, Lambda
from keras.regularizers import l2
from model.attention_layer import Attention
from keras import backend as K


def _bn_relu(x, name=None):
    """
    Helper to build a BN -> relu block
    :param x:
    :param name:
    :return:
    """
    norm = BatchNormalization(name=name+'_bn')(x)
    return Activation("relu", name=name+'_relu')(norm)


def basic_residual_unit(**kwargs):
    """
    Helper to build a BN->relu->conv1d residual unit
    :param res_params:
    :return:
    """

    filters = kwargs['filters']
    b_name = kwargs['name']
    kernel_size = kwargs.setdefault('kernel_size', 16)
    strides = kwargs.setdefault('strides', 1)
    l2r = kwargs['l2r']

    def f(x):
        x = Conv1D(filters=filters,
                   kernel_size=kernel_size,
                   padding='same',
                   strides=strides, kernel_regularizer=l2(l2r) if l2r is not None else None,
                   kernel_initializer='he_normal', name=b_name+'_conv')(x)

        return _bn_relu(x, name=b_name)

    return f


def preact_residual_unit(**kwargs):
    """

    :param kwargs:
    :return:
    """
    filters = kwargs['filters']
    kernel_size = kwargs['kernel_size']
    strides = kwargs['strides']
    drop = kwargs['drop']
    l2r = kwargs['l2r']
    p_name = kwargs['name']

    def f(x):
        x = _bn_relu(x, name=p_name)
        x = Dropout(drop, name=p_name+'_drop')(x)  # the opt place where to put drop
        return Conv1D(filters=filters,
                      kernel_size=kernel_size,
                      padding='same',
                      strides=strides,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2r) if l2r is not None else None,
                      name=p_name+'_conv')(x)  # regularizer will degrade performance

    return f


def residual_block(input_x, idx_block, n_block, **kwargs):
    """

    :param input_x:
    :param idx_block:
    :param n_block:
    :param kwargs:
    :return:
    """
    r_name = kwargs['name']
    init_filters = kwargs['filters']
    filters = init_filters * (2 ** idx_block)
    kernel_size = kwargs['kernel_size']
    strides = kwargs['strides']
    drop = kwargs['drop']
    pool_size = kwargs['pool_size']
    pool_stride = kwargs['pool_stride']
    l2r = kwargs['l2r']

    x = input_x
    x_shortcut = Conv1D(filters=filters, kernel_size=1, name=r_name+'_shortcut_up_conv')(x) if idx_block != 0 else x

    for i in range(n_block):
        pre_name = r_name+'_'+str(i)
        x1 = preact_residual_unit(filters=filters, kernel_size=kernel_size, strides=strides,
                                  drop=drop, l2r=l2r, name=pre_name+'_precat0')(x)
        # x1 = Dropout(drop)(x1)  # if do drop here, then train converge too fast, and lead to overfitting
        x1 = preact_residual_unit(filters=filters, kernel_size=kernel_size, strides=strides,
                                  drop=drop, l2r=l2r, name=pre_name+'_precat1')(x1)

        if (i + 1) % 2 == 0:
            x1 = MaxPooling1D(pool_size=pool_size, strides=pool_stride,
                              name=pre_name+'_poolx_'+str(i))(x1)
            x2 = MaxPooling1D(pool_size=pool_size, strides=pool_stride,
                              name=pre_name+'_poolshort_'+str(i))(x_shortcut)
        else:
            x2 = x_shortcut

        x1 = cse_sse_block(x1, name=pre_name+'_attx')
        x = Add(name=pre_name+'_res_add')([x1, x2])
        x_shortcut = x
        del x1, x2

    return x


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

    se = GlobalAveragePooling1D(name=name+'_gap')(init)
    se = Reshape(se_shape, name=name+'_reshape')(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False, name=name+'_dense_relu')(se)
    se = Dense(filters, activation='sigmoid', use_bias=False, name=name+'_dense_sigmoid')(se)

    x = Multiply(name=name+'_multiply')([init, se])
    return x


def spatial_se_block(ses_input, name=None):
    """
    Create a spatial squeeze-excite block
    Ref:
        [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks]
    (https://arxiv.org/abs/1803.02579)
    :param ses_input:
    :param name
    :return:
    """

    se = Conv1D(1, 1, activation='sigmoid', use_bias=False, name=name+'_conv')(ses_input)
    x = Multiply(name=name+'_multiply')([ses_input, se])
    return x


def cse_sse_block(se_input, ratio=16, name=None):
    """
    Create a spatial squeeze-excite block
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks]
        (https://arxiv.org/abs/1803.02579)
    :param se_input:
    :param ratio:
    :param name
    :return:
    """

    cse = se_block(se_input, ratio, name=name+'_se')
    sse = spatial_se_block(se_input, name=name+'_sse')

    x = Add(name=name+'_sse_add')([cse, sse])
    return x


def simple_residual_block(input_x, **kwargs):
    filters = kwargs['filters']
    kernel_size = kwargs.setdefault('kernel_size', 16)
    strides = kwargs.setdefault('strides', 1)
    drop = kwargs.setdefault('drop', 0.2)
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
    x1 = _bn_relu(x1)
    x1 = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(x1)

    # right branch
    x2 = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(input_x)

    x1 = cse_sse_block(x1)
    x = keras.layers.add([x1, x2])
    del x1, x2
    return x


def cnv_net(win_size, n_in_feat, n_out_class, filters=32, kernel_size=16, strides=1, pool_size=2,
            pool_stride=2, drop=0.5, blocks=(4, 4, 3), fc_size=128,
            kernel_regular_l2=1e-4, temperature=5, m_name='cnvnet'):  #
    """
    :param win_size:
    :param n_in_feat:
    :param n_out_class:
    :param filters:
    :param kernel_size: this would affect the model result
    :param strides:
    :param pool_size:
    :param pool_stride:
    :param drop:
    :param blocks:
    :param fc_size:
    :param kernel_regular_l2:
    :param m_name
    :return:
    """
    input_x = Input(shape=(win_size, n_in_feat), name='input')
    x = basic_residual_unit(filters=filters, kernel_size=kernel_size, l2r=kernel_regular_l2,
                            strides=strides, name='start_basic_res_blk')(input_x)

    for j, n_block in enumerate(blocks):
        x = residual_block(x, j, n_block, filters=filters, kernel_size=kernel_size, strides=strides,
                           pool_size=pool_size, pool_stride=pool_stride, drop=drop,
                           l2r=kernel_regular_l2, name='main_res_blk_'+str(j))
    x = _bn_relu(x, name='end')

    if m_name == 'cnvnet_fc':
        x = Flatten(name='end_flatten')(x)
        x = Dense(fc_size, activation='relu', name='end_dense_relu')(x)
    else:
        x = GlobalAveragePooling1D(name='end_last_global_avg_pool')(x)

    # using temperature to calibrate the output probabilty
    logits = Dense(n_out_class, name='logits')(x)
    logits_t = Lambda(lambda l: l / temperature, name='logit_temperature')(logits)

    out = Activation('softmax', name='model_out_prob')(logits_t)

    model = Model(inputs=input_x, outputs=out, name=m_name)
    return model


def cnv_net_seq(win_size, n_in_feat, n_out_class, filters=32, kernel_size=16, strides=1, pool_size=2,
             pool_stride=2, drop=0.2, blocks=(4, 4, 3), fc_size=128, kernel_regular_l2=None, m_name=None):
    """
    https://www.kaggle.com/sanket30/cudnnlstm-lstm-99-accuracy
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
        :param fc_size:
        :param kernel_regular_l2:
        :param m_name
        :return:
        """
    input_x = Input(shape=(win_size, n_in_feat), name='input')
    # cudnn version >7.3
    x = Bidirectional(CuDNNLSTM(n_in_feat, return_sequences=True),
                      merge_mode='concat')(input_x)

    x = basic_residual_unit(filters=filters, kernel_size=kernel_size,
                            strides=strides, name='start_basic_res_blk')(x)

    for j, n_block in enumerate(blocks):
        x = residual_block(x, j, n_block, filters=filters, kernel_size=kernel_size, strides=strides,
                           pool_size=pool_size, pool_stride=pool_stride, drop=drop,
                           l2r=kernel_regular_l2, name='main_res_blk_' + str(j))
    x = _bn_relu(x, name='end')
    # x = GlobalAveragePooling1D()(x)  #lead to overfitting
    # x = Flatten(name='end_flatten')(x)
    # x = Dense(fc_size, activation='relu', name='end_dense0_relu')(x)
    # x = Dense(fc_size, activation='relu', name='end_dense1_relu')(x)

    x = Attention(x._keras_shape[1])(x)

    out = Dense(n_out_class, activation='softmax', name='end_dense_softmax')(x)
    model = Model(inputs=input_x, outputs=out, name=m_name)
    return model


def cnv_seq(win_size, n_in_feat, n_out_class, filters=64, kernel_size=16, strides=1, pool_size=2,
             pool_stride=2, drop=0.2, blocks=(4, 3), fc_size=128, kernel_regular_l2=None, m_name=None):
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
        :param fc_size:
        :param kernel_regular_l2:
        :param m_name
        :return:
        """
    input_x = Input(shape=(win_size, n_in_feat), name='input')
    x = BatchNormalization()(input_x)

    x = Bidirectional(LSTM(n_in_feat, return_sequences=True, dropout=drop, recurrent_dropout=drop),
                      merge_mode='concat')(x)
    # need cudnn version >7.3
    x = Bidirectional(CuDNNLSTM(n_in_feat, return_sequences=True), merge_mode='concat')(x)

    x_att = Attention(x._keras_shape[-2])(x)
    x = Multiply()([x_att, x])
    x = K.sum(x, axis=1)

    out = Dense(n_out_class, activation='softmax', name='end_dense_softmax')(x)
    model = Model(inputs=input_x, outputs=out, name=m_name)
    return model


def cnv_simple_net(win_size, n_in_feat, n_out_class, filters=64, kernel_size=16, strides=1, pool_size=2,
                   pool_stride=2, drop=0.2):  #
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
    x = Dense(128, activation='relu', )(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(n_out_class, activation='softmax')(x)
    model = Model(inputs=input_x, outputs=out)
    return model


def cnv_aenet(win_size, n_in_feat, filters=64, kernel_size=128,
              strides=1, pool_size=2, pool_stride=2, drop=0.5, n_res_blks=3,
              kernel_regular_l2=1e-4, m_name='cnv_aenet'):

    input_x = Input(shape=(win_size, n_in_feat), name='ae_input')

    x = basic_residual_unit(filters=filters, kernel_size=kernel_size, l2r=kernel_regular_l2,
                            strides=strides, name='encoder_start_blk')(input_x)

    for i in range(n_res_blks):
        x = ae_residual_block(x, filters=filters, kernel_size=kernel_size, strides=strides,
                              pool_size=pool_size, pool_stride=pool_stride, l2r=kernel_regular_l2,
                              drop=drop, name='encoder_resnet_blk_' + str(i))

    x = AveragePooling1D(name='encoder_avgpool')(x)
    encoder = Conv1D(filters=n_in_feat,
                     kernel_size=kernel_size,
                     padding='same',
                     strides=strides,
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(kernel_regular_l2),
                     name='encoder_output')(x)

    d = basic_residual_unit(filters=filters, kernel_size=kernel_size, l2r=kernel_regular_l2,
                            strides=strides, name='decoder_start_blk')(encoder)

    for i in range(n_res_blks):
        d = ae_upsample_res_blk(d, filters=filters, kernel_size=kernel_size, strides=strides, l2r=kernel_regular_l2,
                                pool_size=pool_size, drop=drop, name='decoder_resnet_blk_' + str(i))

    decoded = Conv1D(filters=n_in_feat,
                     kernel_size=kernel_size,
                     padding='same',
                     strides=strides,
                     kernel_regularizer=l2(kernel_regular_l2),
                     kernel_initializer='he_normal',
                     name='decoder_last_conv')(d)
    decoded = UpSampling1D(size=pool_size, name='decoder_last_upsample')(decoded)
    out = Activation('sigmoid', name='decoder_last_prob_output')(decoded)
    ae_model = Model(input_x, out, name=m_name)

    return ae_model


def ae_upsample_res_blk(input_x, **kwargs):
    filters = kwargs['filters']
    kernel_size = kwargs.setdefault('kernel_size', 16)
    strides = kwargs.setdefault('strides', 1)
    drop = kwargs.setdefault('drop', 0.2)
    pool_size = kwargs['pool_size']
    l2r = kwargs['l2r']
    name = kwargs['name']

    # left branch
    x1 = Conv1D(filters=filters,
                kernel_size=kernel_size,
                padding='same',
                strides=strides,
                kernel_regularizer=l2(l2r),
                kernel_initializer='he_normal')(input_x)
    x1 = Dropout(drop)(x1)
    x1 = _bn_relu(x1, name=name + '_br_0')
    x1 = Conv1D(filters=filters,
                kernel_size=kernel_size,
                padding='same',
                strides=strides,
                kernel_regularizer=l2(l2r),
                kernel_initializer='he_normal')(x1)
    x1 = Dropout(drop)(x1)
    x1 = _bn_relu(x1, name=name + '_br_1')
    x1 = UpSampling1D(size=pool_size)(x1)

    # right branch
    x2 = UpSampling1D(size=pool_size)(input_x)

    x = keras.layers.add([x1, x2])
    del x1, x2
    return x


def ae_residual_block(input_x, **kwargs):
    filters = kwargs['filters']
    kernel_size = kwargs.setdefault('kernel_size', 16)
    strides = kwargs.setdefault('strides', 1)
    drop = kwargs.setdefault('drop', 0.2)
    pool_size = kwargs['pool_size']
    pool_stride = kwargs['pool_stride']
    l2r = kwargs['l2r']
    name = kwargs['name']

    # left branch
    x1 = Conv1D(filters=filters,
                kernel_size=kernel_size,
                padding='same',
                strides=strides,
                kernel_regularizer=l2(l2r),
                kernel_initializer='he_normal')(input_x)
    x1 = Dropout(drop)(x1)
    x1 = _bn_relu(x1, name=name + '_br_0')
    x1 = Conv1D(filters=filters,
                kernel_size=kernel_size,
                padding='same',
                strides=strides,
                kernel_regularizer=l2(l2r),
                kernel_initializer='he_normal')(x1)
    x1 = Dropout(drop)(x1)
    x1 = _bn_relu(x1, name=name + '_br_1')
    x1 = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(x1)

    # right branch
    x2 = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(input_x)

    x = keras.layers.add([x1, x2])
    del x1, x2
    return x




