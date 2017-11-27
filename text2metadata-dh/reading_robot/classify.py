# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 07:37:36 2017

@author: jose
"""


"""
To continue developing this script you need:
import pandas as pd
data_train = pd.read_csv("data/data_train.csv", sep="\t", encoding="utf-8", index_col=0)
data_eval = pd.read_csv("data/data_eval.csv", sep="\t", encoding="utf-8", index_col=0)
labels_train = pd.read_csv("data/labels_train.csv", sep="\t", encoding="utf-8", index_col=0, header=None)
labels_eval = pd.read_csv("data/labels_eval.csv", sep="\t", encoding="utf-8", index_col=0, header=None)
print(data_train.shape, data_eval.shape, labels_train.shape, labels_eval.shape)

classifier, training_score, evaluation_score  = classify(data_train, data_eval, labels_train, labels_eval, method = "SVC")

print(training_score, evaluation_score)
# TODO: these files do not represent anymore the data we need for that

"""

from mytoolbox.reading_robot import visualising
from sklearn.model_selection import cross_val_score


def choose_classifier(method = "SVC"):

    if method == "SVC":
        from sklearn.svm import SVC
        classifier = SVC(kernel="linear") # Default C = 1 # , C=100000000000000

    elif method == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=3)

    elif method == "DT":
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier()
                
    elif method == "RF":
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier()
        classifier = BaggingClassifier(tree)

    elif method == "LR":
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression()#C=100000000000000

    elif method == "BN":
        from sklearn.naive_bayes import BernoulliNB
        classifier = BernoulliNB()

    elif method == "GN":
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        """
    # This method doesn't accept negative values in the features, so it breaks with zscores
    # We could move the center of the zscores so that all values are positive
    # Although if we use other features, we would need to modify it in a different way...

    elif method == "MN":
        from sklearn.naive_bayes import MultinomialNB
        classifier = MultinomialNB()
        """
    else:
        print("You haven't chosen a valide classifier!!! You ")
    print("method used:\t", method)   
    return classifier
    
def classify_standard(data_train, data_eval, labels_train, labels_eval, classifier):
    classifier.fit(data_train, labels_train)
    training_score = classifier.score(data_train, labels_train)
    evaluation_score = classifier.score(data_eval, labels_eval)
    
    return training_score, evaluation_score  

def classify_cross(data, labels, classifier, cv = 10):
    scores = cross_val_score(classifier, data, labels, cv = cv) # manuell die samples: k-fold

    return scores  

def make_confusion_matrix(classifier, data_train, data_eval, labels_train, labels_eval, class_, wdir):

    labels_pred = classifier.fit(data_train, labels_train).predict(data_eval)
    print(list(zip(labels_eval.values,labels_pred)))

    class_names = class_names = sorted(list(set(labels_train)))
    
    # Plot non-normalized confusion matrix
    visualising.plot_confusion_matrix(labels_eval, labels_pred, class_values = class_names, class_ = class_, title="Confusion matrix of "+ class_ +" without normalization", wdir = wdir)
    
    # Plot normalized confusion matrix
    visualising.plot_confusion_matrix(labels_eval, labels_pred, class_values=class_names, normalize=True, class_ = class_, title='Normalized confusion matrix of' + class_, wdir = wdir)
    
# TODO: return the values of the classifier.coef_ for the features, for example from SVC, LR, DT or RF

