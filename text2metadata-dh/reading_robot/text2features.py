# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:15:17 2017

@author: jose 

This script converts a Bag of Words model into something different:


"""
"""

# In Spyder

To develope this script in Spyder you need a corpus:

import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("/home/jose/Dropbox/MTB/investigacion/mytoolbox/"))
from reading_robot import load_data
corpus = load_data.corpus2table(
wdir = "/home/jose/cligs/experiments/20170725 reading robot/" ,
wsdir = "corpus/",
verbose = True, min_MFF = 0, max_MFF = 5000,  normalized = False)

"""
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd

def calculate_tfidf(corpus):
    tf_transformer = TfidfTransformer(use_idf=True).fit(corpus)
    X_train_tf = tf_transformer.transform(corpus)
    corpus = pd.DataFrame(X_train_tf.toarray(), columns=corpus.columns, index = corpus.index)
    return corpus

def calculate_relative_frequencies(corpus):
    corpus = corpus.loc[:].div(corpus.sum(axis='columns'), axis="index")
    return corpus

def calculate_zscore(corpus):
    corpus = corpus.loc[:].div(corpus.sum(axis='columns'), axis="index")
    means = corpus.mean(axis = "index")
    stds = corpus.std(axis = "index")   
    corpus = (corpus - means) / stds
    return corpus

def choose_features(corpus, text_representation):
    if text_representation == "raw":
        # If we need to convert the corpus into something different, it would be here
        data = corpus
    elif  text_representation == "relative":
        data = calculate_relative_frequencies(corpus)
    elif  text_representation == "tfidf":
        data = calculate_tfidf(corpus)
    elif  text_representation == "zscores":
        data = calculate_zscore(corpus)
    print(data.head())
    print("textual representation: ", text_representation)
    
    print(data.columns[data.isnull().any()].tolist())
    return data

