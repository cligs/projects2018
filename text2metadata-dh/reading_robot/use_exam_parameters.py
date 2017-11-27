# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:08:27 2017

@author: jose


Example of how to use it:
results_cross, results_mean, results_std, baseline = use_exam_parameters(
wdir = "/home/jose/Dropbox/MTB/investigacion/mytoolbox/reading_robot/data/",
sep =",",
class_ = "author-name",
sampling_mode = "cross",
)
"""
import sys
import os
sys.path.append(os.path.abspath("/home/jose/Dropbox/MTB/investigacion/"))
import pandas as pd

from mytoolbox.reading_robot import load_data, sampling, classify, text2features, cull_data
from scipy import stats


def exam_algorithms_models(wdir, document_data_raw , labels, baseline, least_frequent_class_value, class_, verbose = False, 
    text_representations=["raw","relative","tfidf", "zscores"],
    methods = ["SVC", "KNN", "RF", "DT", "LR", "BN", "GN"],
    sampling_mode = "cross",
    ):
    print("* Examing different classifiers and lexical representations")
    # An empty dataframe is created
    results_mean = pd.DataFrame(columns = text_representations, index = methods)
    results_std = pd.DataFrame(columns = text_representations, index = methods)
    results_cross = pd.DataFrame(columns = text_representations, index = methods)
    print(document_data_raw.shape)
    print(labels.shape)
    for text_representation in text_representations:
        print(text_representation)
        # The corpus is modeled somehow (raw, relative frequencies, tf-idf, z-scores...)
        document_data_model = text2features.choose_features(document_data_raw, text_representation)

        for method in methods:

            classifier = classify.choose_classifier(method = method)

            if sampling_mode == "standard":
                print("* Standard sampling")
                # Data and labels are splitted into train, evaluation and a baseline is created
                test_size = cull_data.calculate_test_size(least_frequent_class_value, verbose = True)
                data_train, data_eval, labels_train, labels_eval = sampling.standard_sampling(document_data_model, labels, test_size = test_size, verbose = verbose) #[class_])
                # Trainging and evaluation are created
                training_score, evaluation_score = classify.classify_standard(data_train, data_eval, labels_train, labels_eval, classifier)#, class_ = class_ )
                results_cross.set_value(method, text_representation, evaluation_score)
                results_cross = results_mean.copy()

            elif sampling_mode == "cross":
                print("* Cross validation sampling")
                cv = cull_data.calculate_cv(least_frequent_class_value)
                print(least_frequent_class_value)
                evaluation_scores = classify.classify_cross(document_data_model, labels, classifier, cv = cv)
                results_cross.set_value(method, text_representation, evaluation_scores)
                results_mean.set_value(method, text_representation, evaluation_scores.mean())
                results_std.set_value(method, text_representation, evaluation_scores.std())
                print("Accuracy: %0.2f (+/- %0.2f)" % (evaluation_scores.mean(), evaluation_scores.std() * 2))
            else:
                print("Your sampling mode is not valid")

    results_cross.to_csv(wdir+"results/results_samp_"+sampling_mode+"_"+class_+"_baseline="+str(baseline)[0:4]+"_methods_models.tsv", sep="\t", encoding="UTF-8")
    results_mean.to_csv(wdir+"results/results_mean_samp_"+sampling_mode+"_"+class_+"_baseline="+str(baseline)[0:4]+"_methods_models.tsv", sep="\t", encoding="UTF-8")
    results_std.to_csv(wdir+"results/results_std_samp_"+sampling_mode+"_"+class_+"_baseline="+str(baseline)[0:4]+"_methods_models.tsv", sep="\t", encoding="UTF-8")

    print(results_mean)
    print(results_cross)
    print("baseline: ", baseline)
    print("\n", class_)

    return results_cross, results_mean, results_std
    
def get_highest_value(results):
    max_value_representation  = results.max(axis="index").sort_values(ascending=False).index[0]
    max_value_method = results.max(axis="columns").sort_values(ascending=False).index[0]
    max_value = results.values.max()
    print("Best results: ", max_value, max_value_representation, max_value_method) 
    
    return max_value, max_value_representation, max_value_method

def test_ttest_cross_results_baseline(results_cross, baseline):

    test_result = stats.ttest_1samp(results_cross, baseline)

    print("result of comparing cross-validation to baseline", test_result)
    return test_result

def test_normality_cross(evaluation_scores):
    """
    """
    print("* testing normality")
    print(evaluation_scores)
    statistic, pvalue = stats.mstats.normaltest(evaluation_scores)
    
    if(pvalue < 0.055):
        result  = "Not normal distribution"
    else:
        result  = "Normal"
    return result
    
def use_exam_parameters(wdir, sep ="\t", verbose=False, class_="class", sampling_mode = "standard"):
    # TODO: Generalize to compare different parameters

    # We load the corpus dataframe (texts as rows, words as columns)
    corpus = load_data.load_corpus(wdir = wdir, freq_table = "freq_table_raw.csv", sep = sep, verbose = verbose, min_MFF = 0, max_MFF = 5000)
    metadata = load_data.load_metadata(wdir = wdir,  sep = sep, verbose = verbose)

    # The metadata of the class is selected and prepared as label 
    document_data_raw, labels, baseline, least_frequent_class_value = cull_data.cull_data(corpus, metadata, class_, verbose)

    # The ML is applied
    results_cross, results_mean, results_std = exam_algorithms_models(wdir = wdir, document_data_raw = document_data_raw, labels = labels, baseline = baseline, least_frequent_class_value = least_frequent_class_value, class_ = class_, verbose = verbose, sampling_mode =  sampling_mode)

    max_value, max_value_representation, max_value_method = get_highest_value(results_mean)

    test_ttest_cross_results_baseline(results_cross.loc[max_value_method,max_value_representation], baseline)
    print("baseline: ", baseline)

    return results_cross, results_mean, results_std, baseline

