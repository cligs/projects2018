# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 08:19:24 2017

@author: jose
"""

"""
This script takes a path in which there are texts files, it reads it and converts it to a MFW dataframe

Example of how to use it:

corpus = corpus2table(
    wdir ="/home/jose/cligs/experiments/20170508 starting_ml/",
    wsdir = "gold-set/",
)
"""

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
import glob


def load_corpus(wdir, wsdir = "corpus/", freq_table = "", sep = "\t", verbose = True, min_MFF = 0, max_MFF = 5000):
    print("* Loading corpus")
    # We load the corpus dataframe (texts as rows, words as columns)
    if freq_table == "" or freq_table == " ":
        print("* Loading texts from folder")
        corpus = corpus2df( wdir = wdir , wsdir = wsdir, verbose = verbose, min_MFF = min_MFF, max_MFF = max_MFF)
    else:
        print("* Opening table")
        corpus = open_freq_table(wdir = wdir, min_MFF = min_MFF, max_MFF = max_MFF, freq_table = freq_table)
    print("corpus' shape: \t", corpus.shape)
    return corpus


def load_metadata(wdir, metadata_table = "metadata.csv",  sep = "\t", verbose = True):
    metadata_df = pd.read_csv(wdir+metadata_table, encoding="utf-8", sep = sep, index_col=0)
    print("metadata and class shape: \t", metadata_df.shape)
    return metadata_df

def open_file(doc, verbose = True):
    """
    This function opens a file and gives back the name of the file and the text
    """
    file_name  = os.path.splitext(os.path.split(doc)[1])[0]
    #print(file_name) if verbose == True else 0
    with open(doc, "r", errors="replace", encoding="utf-8") as fin:
        text = fin.read()
        fin.close()
    return file_name, text

def corpus2dict(wdir, wsdir, text_format, verbose = True):#, case_sensitive, keep_puntuaction):
    """
    This function creates an empty dictionary and adds all the texts in a corpus (file-name as key, text as value)
    """
    # It creates an empty dictionary in which the names of the files are keys and the texts are the values
    corpus = {}
    # It iterates over the files of the folder
    for doc in glob.glob(wdir+wsdir+"*." + text_format):
        # It opens the files and extract file-names and the text
        file_name, text = open_file(doc, verbose = verbose)
        # It adds both thing into the corpus-dictionary
        corpus[file_name] = text
    # It gives back the corpus-dictionary
    return corpus

def cut_corpus(corpus, min_MFF=0, max_MFF=5000):
    """
    This function cut the amount of MFF to be used
    """
    corpus = corpus.iloc[:, min_MFF : max_MFF]
    return corpus

def corpusdict2df(wdir, corpus, keep_punctuation, save_files=True, normalized = True, verbose = True):
    """
    This function takes the corpus-dictionary and converts it into a dataframe
    """
    # We define how we want the tokeniser
    if keep_punctuation == True:
        vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b|[¶\(»\]\?\.\–\!’•\|“\>\)\-\—\:\}\*\&…¿\/=¡_\"\'·+\{\#\[;­,«~]")
    else:
        vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    # The tokenisator is used
    texts = vec.fit_transform(corpus.values())

    # The output of the tokenisor and the keys are used to create a corpus-dataframe
    corpus = pd.DataFrame(texts.toarray(), columns=vec.get_feature_names(), index=corpus.keys())

    # corpus-dataframe is sorted using the index (i.e. the names of the files)
    corpus = corpus.sort_index(axis="index")

    # It saves the file
    if save_files == True:
        corpus = corpus.T
        corpus["sum"] = corpus.sum(axis="columns")
        corpus = corpus.sort_values(by = "sum", ascending=False)
        del corpus['sum']
        corpus = corpus.T
        corpus.to_csv(wdir+"freq_table_raw.csv", sep='\t', encoding='utf-8', index=True)

    # It converts it to relative frequencies
    # I wonder if this step should be elsewhere. Do we allways want to work with relative frequencies? And what about other conversions like tf-idf or z-scores? """
    # What if instead of using the sum of the frequencies in the corpus, we would use the mean of the frequency in the whole corpus?"""

    print(corpus.head())
    return corpus

def open_freq_table(wdir, min_MFF, max_MFF, freq_table):
    corpus = pd.read_csv(wdir+freq_table, encoding="utf-8", sep="\t", index_col=0)
    print("corpus shape: ", corpus.shape)

    if max_MFF != False:
        print("cuting features of corpus!")
        corpus = cut_corpus(corpus, min_MFF, max_MFF)
    print("corpus shape: ", corpus.shape)

    return corpus

def corpus2df(wdir, wsdir = "corpus/", text_format="txt", keep_punctuation = False, save_files=True, min_MFF = 0, max_MFF = 5000, verbose = True, normalized = True):
    """
    Main function. It takes a wdir with texts and make a dataframe of relative frequencies.
    """
    print("reading texts!")
    # First, we open the files and convert them into a dictionary (file-names as keys, texts as values)
    corpus = corpus2dict(wdir, wsdir, text_format, verbose = verbose)

    print("corpus with: \t", len(corpus), " samples ")

    # Second, we convert it to a dataframe with the relative frequency
    corpus = corpusdict2df(wdir, corpus, keep_punctuation, save_files, verbose = verbose, normalized = normalized)
    #print(corpus.index)

    if max_MFF != False:
        print("cuting features of corpus!")
        
        # Third, we take only as much features as we want
        corpus = cut_corpus(corpus, min_MFF, max_MFF)
        print("final corpus shape: ", corpus.shape)

    return corpus
