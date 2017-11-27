# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 07:37:36 2017

@author: jose
"""

 
from sklearn.model_selection import train_test_split
from collections import Counter

    
def standard_sampling(data, labels, test_size, verbose = True):
    print("* Sampling data in standard way")
    data_train, data_eval, labels_train, labels_eval = train_test_split(data, labels, test_size = test_size, stratify=labels) #, random_state = 0 )
    print(data_train.shape, data_eval.shape, labels_train.shape, labels_eval.shape) if verbose == True else 0
    print(Counter(labels_train), Counter(labels_eval)) if verbose == True else 0
    
    return data_train, data_eval, labels_train, labels_eval

