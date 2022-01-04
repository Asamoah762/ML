#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 01:13:36 2022

@author: eric
"""

#from preprocessing import Preprocessing
import pickle

class Regression:
    
    # Initialization
    def __init__(self, clsifier):
        self.clsifier = clsifier
        
    # Multiple Regression Model
    def Multiple_RM(X_train, y_train):
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        filename = 'Multiple_RM.sav'
        regressor.fit(X_train, y_train)
        pickle.dump(regressor, open(filename, 'wb'))
        return regressor
         
    # Polynomial Regression Model    
    def Polynomial_RM(X_train, y_train, X_test, degree = 2):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        poly_reg = PolynomialFeatures(degree = degree)
        X_poly = poly_reg.fit_transform(X_train)
        X_test_poly = poly_reg.transform(X_test)
        regressor = LinearRegression()
        regressor.fit(X_poly,y_train)
        filename = 'Polynomial_RM.sav'
        pickle.dump(regressor, open(filename, 'wb'))
        return regressor, X_test_poly
    
    # Support Vector Regression Model
    def SVR_RM(X_train, y_train, kernel = 'rbf'):
        from sklearn.svm import SVR
        regressor = SVR(kernel = kernel)
        regressor.fit(X_train, y_train)
        filename = 'SuportVector_RM.sav'
        pickle.dump(regressor, open(filename, 'wb'))
        return regressor
    
    # Decision Tree Regression Model
    def DecisionTree_RM(X_train, y_train):
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(random_state = 0)
        regressor.fit(X_train, y_train)
        filename = 'DecisionTree_RM.sav'
        pickle.dump(regressor, open(filename, 'wb'))
        return regressor
    
    # Random Forest Regression Model
    def RandomForest_RM(X_train, y_train):
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators= 50, random_state=0)
        regressor.fit(X_train,y_train)
        filename = 'RandomForest_RM.sav'
        pickle.dump(regressor, open(filename, 'wb'))
        return regressor
    
    # Regression and Prediction
    def RegressionModels_Predict(X_train, X_test, y_train):
        # Regression models
        Multi_regressor = Regression.Multiple_RM(X_train, y_train)
        Poly_regressor, X_test_poly = Regression.Polynomial_RM(X_train, y_train, X_test)
        DecisionTree_regressor = Regression.DecisionTree_RM(X_train, y_train)
        RandomForest_regressor = Regression.RandomForest_RM(X_train, y_train)
        SVR_regressor = Regression.SVR_RM(X_train, y_train)
        
        # Predictions
        y_Multi_pred = Multi_regressor.predict(X_test)
        y_Poly_pred = Poly_regressor.predict(X_test_poly)
        y_DecisionTree_pred = DecisionTree_regressor.predict(X_test)
        y_RandomForest_pred = RandomForest_regressor.predict(X_test)
        y_SVR_pred = SVR_regressor.predict(X_test)
        return y_Multi_pred, y_Poly_pred, y_DecisionTree_pred, y_RandomForest_pred, y_SVR_pred
        