# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:52:43 2017

@author: jose
"""
import pandas as pd
def metadata2numbers(metadata_df):
    
    metadata_df = pd.get_dummies(metadata_df)
    
    return metadata_df

