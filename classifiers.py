#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 18:25:33 2021

@author: eric
"""

import sklearn

class Classifier:
    
    # Initialization
    def __init__(self, clsifier):
        #Preprocessing.super().__init__(X_train, y_train)
        self.clsifier = clsifier
    
    # Random Forest Classifier    
    def RFC(X_train, y_train, **kwargs):
        from sklearn.ensemble import RandomForestClassifier
        clfier = RandomForestClassifier(**kwargs)
        return clfier.fit(X_train, y_train)
    
    # K Nearest Neighbour classifier    
    def kNN(X_train, y_train, **kwargs):
        clfier = sklearn.neighbors.KNeighborsClassifier(**kwargs)
        return clfier.fit(X_train, y_train)
    
    # Logistic Regression classifier    
    def LR(X_train, y_train, **kwargs):
        clfier = sklearn.linear_model.LogisticRegression(**kwargs)
        return clfier.fit(X_train, y_train)
        
    # Naive Bayes classifier
    def GaussNB(X_train, y_train):
        from sklearn.naive_bayes import GaussianNB
        clfier = GaussianNB()
        return clfier.fit(X_train, y_train)
    
    # Decision Tree classifier    
    def DTC(X_train, y_train, **kwargs):
        clfier = sklearn.tree.DecisionTreeClassifier(**kwargs)
        return clfier.fit(X_train, y_train)
        
    # Support Vector Machine classifier    
    def SuppVM(X_train, y_train, **kwargs):
        clfier = sklearn.svm.SVC(**kwargs)
        return clfier.fit(X_train, y_train)