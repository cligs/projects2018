# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:15:05 2017

@author: jose
"""
import sys
import os
sys.path.append(os.path.abspath("/home/jose/Dropbox/MTB/investigacion/"))
from mytoolbox.reading_robot import supervised

results = supervised.supervised(
wdir ="/home/jose/cligs/experiments/20170725 reading robot/",
sep = ",",
freq_table = "freq_table_raw.csv",
verbose=False,
text_representation = "tfidf",
method = "SVC",
classes = ["narrator"],
)