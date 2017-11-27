# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:38:36 2017

@author: jose


import sys
import os
sys.path.append(os.path.abspath("/home/jose/cligs/"))
sys.path.append(os.path.abspath("/home/jose/Dropbox/MTB/investigacion/"))

from mytoolbox.reading_robot import use_ml_rules

definitions_rules = [
#["setting-name", {"madrid": "Madrid"},{"bilbao" : "Bilbao","brandeso" : "Brandeso","castellar" : "Castellar","confines" : "Confines","granburgo" : "Granburgo","guadix" : "Guadix","jerez" : "Jerez","joya" : "La Joya","marineda" : "Marineda","parcent" : "Parcent","ulloa" : "Pazos de Ulloa","santander" : "Santander","tablanca" : "Tablanca","valencia" : "Valencia","valverde" : "Valverde de Lucerna","vetusta" : "Vetusta","villaalegre" : "Villaalegre","villaruin" : "Villaruin","Mixed" : "mixed","none" : "n.av.","other_mixed" : "other/mixed","Unknown" : "unknown", "Unknown2" : "?"},0.5],
#["setting-country", {"españa": "Spain"}, {"méxico": "Mexico", "santafe": "Santa Fe de Tierra", "atargea": "Atargea", "atarjea": "Atarjea", "other_word":"mixed"}, 0.9],
["type-end_simp", {"triste": "negative"}, {"feliz": "positive", "neutro": "neutral"}, 0.55],
["narrator", {"dijo": "third-person"}, {"dije": "first-person", "dijiste": "second-person"}, 0.9],
["protagonist-profession", {"párroco" : "religious"}, {"artistas" : "artist","méndigo" : "beggar","dueños" : "business owner","médicos" : "doctor", "esposas" : "housewife","señoritos" : "lord","soldado" : "military personnel","mixed_token" : "mixed","nav_token" : "n.av.","parlamentario" : "politician","funcionario" : "public servant","científico" : "researcher","estudiante" : "student","trabajador" : "worker"}, 0.9],
["protagonist-social-level" , {"puesto": "medium"}, {"posesiones": "high", "miseria" : "low"}, 0.66],
["representation", {"real": "realistic"}, {"ficticio": "non-realistic"}, 1],
["setting-continent",{"españa": "Europe"},{"méxico": "America", "other_word":"mixed"},1],
["protagonist-gender",{"él": "male"},{"ella": "female"},0.5],
["setting",{"pueblos": "rural"},{"capital": "big-city", "provincia": "small-city", "barco":"boat"},0.6],
["protagonist-age",{"crecido": "adult"},{"niño": "child", "anciano" : "mature", "joven": "young"},0.25],
["time-period",{"actualidad": "contemporary"},{"renacimiento": "modern_times"},0],
]

definitions_rules = [
["setting-continent",{"españa": "Europe"},{"méxico": "America", "other_word":"mixed"},1],
]
df_rules_results = use_ml_rules_all(wdir = "/home/jose/Dropbox/MTB/investigacion/mytoolbox/reading_robot/data/", sep = ",", definitions_rules = definitions_rules, min_MFF = 0, max_MFF = 100,)
"""

import sys
import os
sys.path.append(os.path.abspath("/home/jose/Dropbox/MTB/investigacion/"))
import pandas as pd
from scipy import stats


from mytoolbox.reading_robot import calculating_rules, load_data, use_exam_parameters, cull_data

def set_value_df_rules_result(df_rules_results, class_, baseline, rules_score, f1_cross, f1_cross_mean, f1_cross_stdev, algorithm, model):
    df_rules_results.set_value( class_, "f1_rule", rules_score)
    df_rules_results.set_value( class_, "f1_baseline", baseline)
    df_rules_results.set_value( class_, "f1_cross", f1_cross)
    df_rules_results.set_value( class_, "f1_cross_mean", f1_cross_mean)
    df_rules_results.set_value( class_, "f1_cross_stdev", f1_cross_stdev)
    df_rules_results.set_value( class_, "algorithm", algorithm)
    df_rules_results.set_value( class_, "model", model)

    if rules_score > baseline:
        is_rule_over_base = 1
    else:
        is_rule_over_base = 0
    df_rules_results.set_value( class_, "is_rule_over_base", is_rule_over_base)


    statistic, pvalue_ttest1samp_rules_base = stats.ttest_1samp(f1_cross,baseline)
    if f1_cross_mean > baseline and pvalue_ttest1samp_rules_base < 0.055:
        is_cross_over_base = 1
    else:
        is_cross_over_base = 0
        
    df_rules_results.set_value( class_, "is_cross_over_base", is_cross_over_base)
            
    statistic, pvalue_ttest1samp_rules_base = stats.ttest_1samp(f1_cross,rules_score)
    if f1_cross_mean > rules_score and pvalue_ttest1samp_rules_base < 0.055:
        is_cross_over_rule = 1
    else:
        is_cross_over_rule = 0
    df_rules_results.set_value( class_, "is_cross_over_rule", is_cross_over_rule)

    if is_rule_over_base == 1 and is_cross_over_rule == 0:
        winner = "Rule"
    elif is_cross_over_base == 1 and is_cross_over_rule == 1:
        winner = "ML"
    elif is_rule_over_base == 0 and is_cross_over_base == 0:
        winner = "Baseline"
    elif is_rule_over_base == 0 and is_cross_over_base == 1 and is_cross_over_rule == 0:
        winner = "ML"
    else:
        winner = "Else"

    df_rules_results.set_value( class_, "winner", winner)

    return df_rules_results

def apply_ml_comparing_rules(df_rules_results, wdir, document_data_raw, labels, verbose, class_, rules_score, baseline, least_frequent_class_value):

    # cross validation
    results_cross, results_mean, results_std = use_exam_parameters.exam_algorithms_models(wdir = wdir, document_data_raw = document_data_raw, labels = labels, baseline = baseline, least_frequent_class_value = least_frequent_class_value, class_ = class_, verbose = verbose, sampling_mode =  "cross")
    # gets maximim values
    max_value, max_value_representation, max_value_method = use_exam_parameters.get_highest_value(results_mean)
    print(max_value, max_value_representation, max_value_method)
    # add to results dataframe
    df_rules_results = set_value_df_rules_result(df_rules_results = df_rules_results, class_ = class_, baseline = baseline, rules_score = rules_score, f1_cross = results_cross.loc[max_value_method,max_value_representation], f1_cross_mean = max_value, f1_cross_stdev = results_std.loc[max_value_method,max_value_representation], algorithm = max_value_method, model = max_value_representation)
    
    return df_rules_results


def use_ml_rules_all(wdir, definitions_rules, wsdir = "corpus/", freq_table = "", metadata_table = "metadata.csv", sep = "\t", verbose = True, min_MFF = 0, max_MFF = 0, ):
    # We load metadata and data
    metadata = load_data.load_metadata(wdir = wdir, metadata_table = metadata_table,  sep = sep, verbose = verbose)
    complet_corpus = load_data.load_corpus(wdir = wdir, wsdir = wsdir, freq_table = freq_table, sep = sep, verbose = verbose, min_MFF = 0, max_MFF = 0)

    corpus = load_data.cut_corpus(complet_corpus, min_MFF= min_MFF, max_MFF = max_MFF)
    classes = [item[0] for item in definitions_rules]
    print(classes)

    # Create a dataframe for the results
    df_rules_results = pd.DataFrame(columns=["f1_baseline","f1_rule","f1_cross_mean","f1_cross_stdev","algorithm","model","is_rule_over_base","is_cross_over_base","is_cross_over_rule","winner","f1_cross"], index = classes)
    print(df_rules_results)
    for rule in definitions_rules:
        print(rule)
        class_ = rule[0]
        default_feature_class_labels = rule[1]
        non_default_features_class_labels = rule[2]
        default_features_threshold = rule[3]

        print("Shape of data for culling class", corpus.shape)
        print(class_, default_feature_class_labels)
        document_data_raw, labels, baseline, least_frequent_class_value = cull_data.cull_data(corpus, metadata, class_, verbose)
        rules_score = calculating_rules.rule_tokens(document_data_raw, labels, default_feature_class_labels = default_feature_class_labels, non_default_features_class_labels = non_default_features_class_labels, class_ = class_, default_features_threshold = default_features_threshold, wdir = wdir, corpus = complet_corpus)
        document_data_raw = load_data.cut_corpus(document_data_raw, min_MFF= min_MFF, max_MFF = max_MFF)
        df_rules_results = apply_ml_comparing_rules(df_rules_results = df_rules_results, wdir = wdir, document_data_raw = document_data_raw, labels = labels, verbose = verbose, class_ = class_, rules_score = rules_score, baseline = baseline, least_frequent_class_value = least_frequent_class_value)
        print("Final shape of data", document_data_raw.shape)

    print(df_rules_results[["f1_baseline","f1_rule","f1_cross_mean","algorithm","model","winner"]])
    df_rules_results.round(2).to_csv(wdir+"results/results_comp_rules_cross_"+str(max_MFF)+".tsv", sep="\t", encoding="UTF-8")
    
    return df_rules_results

