# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 07:37:36 2017

@author: jose
"""

 
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import f1_score
import pandas as pd

def extract_labels(metadata, class_, verbose = True):
    labels_df = metadata[class_]

    print("labels done")
    return labels_df


def calculate_baseline(labels):
    classes_values = labels.values.tolist()

    # This functions gets a list of the values (from all the metadata or labels) and gives back a baseline for the most-populated class
    counter_labels = Counter(classes_values)
    baseline = sorted(counter_labels.values())[-1] / sum(counter_labels.values()) #baseline_precission

    return baseline


def cull_data_problematic_class_values(counter_labels, data, labels, verbose = True):
    #print("labels!: ",labels)
    problematic_values = []
    for value, freq in counter_labels.items():
        if freq < 2:
            problematic_values.append(value)

    # TODO: esto deber´iamos pasarlo como argumento
    problematic_class_values = ["n.av.", "other", "mixed", "?", "unknown","none", "second-person"]
    problematic_values = problematic_values + problematic_class_values

    print("Problematic values: ", problematic_values)

    problematic_ids = []
    for problematic_value in problematic_values:
        new_problematic_ids = list(labels.loc[labels.values == problematic_value].index)
        problematic_ids = problematic_ids + new_problematic_ids
    print("problematic ids: ",problematic_ids)
    #print((labels))
    #print((data))
    print(labels.shape)
    print(data.shape)
    data = data.drop(problematic_ids)
    labels = labels.drop(problematic_ids)

    print(Counter(list(labels.values)))
    print(data.shape, labels.shape)
    return data, labels


def calculate_test_size(least_frequent_class_value, verbose = True):
    """
    This function takes the labels, makes a Counter, takes the least frequent value and based on that, it defines the size of test set
    """

    if least_frequent_class_value == 2:
        print("Your least frequent class has only 2 members. So the system learns from the half of the examples and apply to the other half. This can make your results worse.")
        test_size = 0.5
    elif least_frequent_class_value == 3:
        print("Your least frequent class has only 3 members. So the system learns from two thirds of he example and apply to the other third. This can make your results worse.")
        test_size = 0.33
    else:
        print("The size of your classes is good.") if verbose == True else 0
        test_size = 0.25
            
    return test_size

def calculate_cv(least_frequent_class_value):
    if least_frequent_class_value >= 10:
        cv = 10
    else:
        cv = least_frequent_class_value
    return cv

def drop_empty_features(data):
    """
        This function deletes the columns (features) that are completly empty.
        Why? A text could contain some words (normally proper names) that doesn't appear in other text
        in the corpus. If we delete this text from the corpus dued its label for a class, then the feature
        has a value of 0 in all the other texts. If we then calculate zscores based on that, it will break
        because we will divide by 0.
        That means that the number of features is relative to the class we are studying (uouh!)
    """
    data = data.loc[:, (data != 0).any(axis=0)]
    return data
    
def cull_data(corpus, metadata, class_, verbose): #  type_least_populated = "least_populated_value"
    print("* Culling data")
    # One class is selected
    labels = extract_labels(metadata = metadata, class_ = class_, verbose = verbose)
    
    counter_labels = Counter(list(labels.values))
    print(counter_labels)
    
    least_frequent_class_value = counter_labels.most_common()[-1][1]

    if least_frequent_class_value < 10:
        print("Some of your class values have less than 10 examples. We are going to cull them.\n")
        data, labels = cull_data_problematic_class_values(counter_labels, corpus, labels, verbose = True)
        data = drop_empty_features(data)
    else:
        print("All your class values had more than 10 examples. Congrats :)")
        data = corpus
        
    least_frequent_class_value = Counter(list(labels.values)).most_common()[-1][1] 
    baseline = calculate_baseline(labels)

    print("Least popupated class value:\t ", least_frequent_class_value) if verbose == True else 0

    print("Labels shape", labels.shape, "\n Data shape",data.shape)
    return data, labels, baseline, least_frequent_class_value
