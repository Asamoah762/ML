#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 01:38:58 2022

@author: eric
"""
# Libraries imported
from regressors import Regression
from preprocessing import Preprocessing
import os
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Data Path 
PATH = './data/'
names = os.listdir(PATH)

# Dictionary for Regression Models
Multi_dic = {}
Poly_dic = {}
DecisionTree_dic = {}
RandomForest_dic = {}
SVR_dic = {}
 
for filename in names:
    
    # Directory of the data-file
    file = Preprocessing(PATH + filename)
        
    # Load the data-file
    dataset = file.data(file.datafile)
        
    # Independent and Dependent variable
    X = dataset.iloc[:,1:3].values
    y = dataset.iloc[:,-1].values
    y = y.reshape(len(y), 1)
    
    # Split Dataset into train & test data
    X_train, X_test, y_train, y_test = Preprocessing.datasplit(X,y, test_size = 0.20, random_state = 0)
    
    # Regresion Models & Predictions
    y_Multi_pred, y_Poly_pred, y_DecisionTree_pred, y_RandomForest_pred, y_SVR_pred = Regression.RegressionModels_Predict(X_train, X_test, y_train)
    
    # Compute the r2_score
    r2_score_err_M = r2_score(y_test,y_Multi_pred)
    r2_score_err_P = r2_score(y_test,y_Poly_pred)
    r2_score_err_DecisionTree = r2_score(y_test, y_DecisionTree_pred)
    r2_score_err_RandomForest = r2_score(y_test, y_RandomForest_pred)
    r2_score_err_SVR = r2_score(y_test, y_SVR_pred)
    
    # Extract file name
    name = filename.split('.')
    keys = name[0].split('-')
    key = keys[1]
    
    # Add the r2-score for each data to a dictionary
    Multi_dic.update({key: r2_score_err_M})
    Poly_dic.update({key: r2_score_err_P})
    DecisionTree_dic.update({key: r2_score_err_DecisionTree})
    RandomForest_dic.update({key: r2_score_err_RandomForest})
    SVR_dic.update({key: r2_score_err_SVR})
    
# r2-score to plot
M_key_list = list(Multi_dic.keys())
M_value_list = list(Multi_dic.values())
P_value_list = list(Poly_dic.values())
DT_values_list = list(DecisionTree_dic.values())
RF_values_list = list(RandomForest_dic.values())
SVR_value_list = list(SVR_dic.values())

# Visualization of r2-score
plt.figure()
plt.plot(M_key_list, M_value_list, color = 'red')
plt.plot(M_key_list, P_value_list, color = 'green')
plt.plot(M_key_list, DT_values_list, color = 'orange')
plt.plot(M_key_list, RF_values_list, color = 'pink')
plt.plot(M_key_list, SVR_value_list, color = 'blue')
plt.xlabel('Days')
plt.ylabel('r2-Score')
plt.legend(('Multi', 'Poly', 'DecisionTree', 'RandomForest'), loc = 'upper right')
plt.tick_params(axis='x', which='major', labelsize=7)
plt.savefig('r2-score-comparison.png')
plt.show()