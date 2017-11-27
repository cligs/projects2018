# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:14:48 2017

@author: jose
"""
"""
This script is the principal from Reading Robot and its function is to use classification ore regression (this last part still to be done) on sets (evaluating or testing).

# Parameters:

* wdir (string) : path to the corpus
* wsdir (string) : path to the subcorpus. Default = "corpus/"
* metadata_table (string) : name of the csv file of corpus. Default ="metadata.csv"
* sep (string) : separator of the CSV file. Default = "\t"
* classes (list of string) : with names of the columns that we want to test = ["class"]
* verbose (boolean) : default = True
* method (string) : method to be used. Possible values: "SVC", "KNN", "RF", "DT", "LR", "BN", "GN". Default = "SVC"
* min_MFF (int) : number of minimum Most Frequent Features. Default = 0
* max_MFF (string) : number of maximum Most Frequent Features. Default = 5000
* text_representation (string) : model for the lexical representation. Possible values: "relative", "raw", "tfidf", "zscores"

# Example of how to use it:

Import first:

import sys
import os
sys.path.append(os.path.abspath("/home/jose/Dropbox/MTB/investigacion/"))

from mytoolbox.reading_robot import use_supervised

data, labels, classifier, evaluation_score, base_line = use_supervised.supervised(
wdir ="/home/jose/Dropbox/MTB/investigacion/mytoolbox/reading_robot/data/", 
sep = ",",
verbose = False,
method = "SVC",
text_representation = "raw",
classes = ["narrator"],
sampling_mode = "standard",
)

results = use_supervised.supervised(wdir ="/home/jose/Dropbox/MTB/investigacion/mytoolbox/reading_robot/data/",  sep = ",", verbose=False)

data, labels, classifier, evaluation_score, base_line = supervised(
wdir ="/home/jose/Dropbox/MTB/investigacion/mytoolbox/reading_robot/data/", 
sep = ",",
verbose = False,
method = "RF",
text_representation = "zscores",
classes = ["narrator"],
sampling_mode = "cross",
)


"""
import sys
import os
sys.path.append(os.path.abspath("/home/jose/Dropbox/MTB/investigacion/"))

import numpy as np

from mytoolbox.reading_robot import load_data, sampling, classify, text2features, visualising, cull_data, use_exam_parameters
from sklearn import svm

def supervised(wdir, wsdir = "corpus/", freq_table = "", metadata_table = "metadata.csv", sep = "\t", classes = ["class"], verbose = True, method = "SVC", min_MFF = 0, max_MFF = 5000, text_representation="relative", sampling_mode = "standard"):
    
    # We load the corpus dataframe (texts as rows, words as columns)
    corpus = load_data.load_corpus(wdir = wdir, wsdir = wsdir, freq_table = freq_table, sep = sep, verbose = verbose, min_MFF = min_MFF, max_MFF = max_MFF)
    # We load the metadata
    metadata = load_data.load_metadata(wdir = wdir, metadata_table = metadata_table,  sep = sep, verbose = verbose)
        
    print(corpus.shape)
    print(metadata.shape)
    
    # For each element in the classes list we classify the information:
    for class_ in classes:
        print("\n\nanalysed class:\t", class_)

        document_data_raw, labels, baseline, least_frequent_class_value = cull_data.cull_data(corpus, metadata, class_, verbose)

        # The corpus is modeled somehow (raw, relative frequencies, tf-idf, z-scores...)
        document_data_model = text2features.choose_features(document_data_raw, text_representation)
        #print(document_model_data)

        classifier = classify.choose_classifier(method = method)

        if sampling_mode == "standard":
            print("standard sampling")
            test_size = cull_data.calculate_test_size(least_frequent_class_value, verbose = True)
            
            # Data and labels are splitted into train, evaluation
            data_train, data_eval, labels_train, labels_eval = sampling.standard_sampling(document_data_model, labels, test_size, verbose = verbose) #[class_])

            # Trainging and evaluation are created
            training_score, evaluation_score = classify.classify_standard(data_train, data_eval, labels_train, labels_eval, classifier = classifier)#, class_ = class_ )
            scores = training_score, evaluation_score
            classify.make_confusion_matrix(classifier = classifier, data_train = data_train, data_eval = data_eval, labels_train = labels_train, labels_eval = labels_eval, class_ = class_, wdir = wdir)
        elif sampling_mode == "cross":
            cv = cull_data.calculate_cv(least_frequent_class_value)
            print("cross validation sampling")
            scores = classify.classify_cross(document_data_model, labels, classifier, cv = cv)
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            test_result = use_exam_parameters.test_ttest_cross_results_baseline(scores, baseline)
            print(test_result)
        
        print("Scores: \t", scores)
        print("Baseline: \t\t", baseline)
            
    return document_data_model, labels, classifier, scores, baseline