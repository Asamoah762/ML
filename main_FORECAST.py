#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 01:38:58 2022

@author: eric
"""

from regressors import Regression
from preprocessing import Preprocessing
import os
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

PATH = 'DIRECTROY OF THE FILE'
names = os.listdir(PATH)

Multi_dic = {}
Poly_dic = {}
 
for filename in names:
    
    # Directory of the data-file
    file = Preprocessing(PATH + filename)
        
    # Load the data-file
    dataset = file.data(file.datafile)
        
    # Independent and Dependent variable
    X = dataset.iloc[:,1:3].values # SELECT INDEPENDENT VARIABLES // NOTE: THE SIZE OF INDEPENDENT VARIABLE DEPENDS ON THE DATA.
    y = dataset.iloc[:,-1].values # SELECT DEPEENDENT VARIABLES // NOTE: THE SIZE OF DEPENDENT VARIABLE DEPENDS ON THE DATA.
    y = y.reshape(len(y), 1)
    
    # Split Dataset into train & test data
    X_train, X_test, y_train, y_test = Preprocessing.datasplit(X,y, test_size = 0.15, random_state = 0)
    
    # Regression models
    Multi_regressor = Regression.Multiple_RM(X_train, y_train)
    Poly_regressor, X_test_poly = Regression.Polynomial_RM(X_train, y_train, X_test)
    
    # Predictions
    y_Multi_pred = Multi_regressor.predict(X_test)
    y_Poly_pred = Poly_regressor.predict(X_test_poly)
    
    # Compute the r2_score
    r2_score_err_M = r2_score(y_test,y_Multi_pred)
    r2_score_err_P = r2_score(y_test,y_Poly_pred)
    
    # Extract file name
    name = filename.split('.')
    keys = name[0].split('-')
    key = keys[1]
    
    # Add the r2-score for each data to a dictionary
    Multi_dic.update({key: r2_score_err_M})
    Poly_dic.update({key: r2_score_err_P})
    
# r2-score to plot
M_key_list = list(Multi_dic.keys())
M_value_list = list(Multi_dic.values())
P_value_list = list(Poly_dic.values())

# Visualization of r2-score
plt.figure()
plt.plot(M_key_list, M_value_list, color = 'red')
plt.plot(M_key_list, P_value_list, color = 'green')
plt.xlabel('Days')
plt.ylabel('r2-Score')
plt.legend(('Multi', 'Poly'), loc = 'upper right')
plt.tick_params(axis='x', which='major', labelsize=7)
plt.savefig('r2-score-comparison.png')
plt.show()


