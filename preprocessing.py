#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 18:30:00 2021

@author: eric
"""
"""
This data preprocessing template is a basic template in Machine Learning
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocessing:

    
    # Initialization
    def __init__(self, datafile):
        self.datafile = datafile # csv,tsv, etc file
        
    # function to load and assign data to their corresponding independent and dependent variables
    def data(self, datafile , **kwargs):
        return  pd.read_csv(self.datafile, **kwargs)

    # function to split data into training and testing set
    def datasplit(X, y, **kwargs):
        X_train, X_test, y_train, y_test =\
        train_test_split(X, y, **kwargs)
        return X_train, X_test, y_train, y_test 
    
    # function to standardize features by removing the mean and scaling to unit variance   
    def StdScaler(X_train, y_train):
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        y_train = sc.fit_transform(y_train)
        return X_train, y_train
        