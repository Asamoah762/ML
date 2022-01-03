#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 01:13:36 2022

@author: eric
"""

import sklearn

class Regression:
    
    # Initialization
    def __init__(self, clsifier):
        self.clsifier = clsifier
        
    # Multiple Regression Model
    def Multiple_RM(X_train, y_train):
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        return regressor.fit(X_train, y_train)
        
    def Polynomial_RM(X_train, y_train, X_test, degree = 2):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        poly_reg = PolynomialFeatures(degree = degree)
        X_poly = poly_reg.fit_transform(X_train)
        X_test_poly = poly_reg.transform(X_test)
        regressor = LinearRegression()
        return regressor.fit(X_poly,y_train), X_test_poly
        