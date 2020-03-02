#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: __init__.py.py
    Description:
    
Created by Yong Bai on 2019/8/20 2:45 PM.
"""
from .load_data import CNVDataGenerator, CNVDataGenerator2, CNVDataGenerator3, CNVDataGenerator_AE
from .train_run import train, train2, train3, train_ae
from .evaluate_run import evaluate
# from .cv_train import cv_train
# from .evaluate_run import evaluate
# from .multinodes_cvtrain import mnd_train
