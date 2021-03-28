#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 23:25:24 2021
@author: eric
"""

# Import libraries
from classifiers import Classifier
from preprocessing import Preprocessing
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

file = Preprocessing('Restaurant_Reviews.tsv')
dataset = file.data(file.datafile, delimiter = '\t', quoting = 3)

nltk.download('stopwords')
corpus = []
sw_list = ["not","isn't","doesn't","hasn't","hadn't","don't", "couldn't", "haven't", "wouldn't"]

# Length of dataset
N = len(dataset)
n = len(sw_list)-1

for i in range(0, N):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    for j in range(0,n):
        all_stopwords.remove(sw_list[j])
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

# Spliting of dataset
X_train, X_test, y_train, y_test = Preprocessing.datasplit(X,y, test_size = 0.1, random_state = 0)

# Classifiers
classifierRFC = Classifier.RFC(X_train, y_train, n_estimators = 13, criterion = 'entropy')
classifierkNN = Classifier.kNN(X_train, y_train, n_neighbors = 8, metric = 'minkowski')
classifierLR = Classifier.LR(X_train, y_train)
classifierGaussNB = Classifier.GaussNB(X_train, y_train)
classifierDTC = Classifier.DTC(X_train, y_train, criterion = 'entropy')
classifierSVM = Classifier.SuppVM(X_train, y_train, kernel = 'rbf')

# Prediction using the test set
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


""" Random Forest Classifier """
y_predRFC = classifierGaussNB.predict(X_test)
cm_RFC = confusion_matrix(y_test, y_predRFC)
ac_RFC = accuracy_score(y_test, y_predRFC)*100
pc_RFC = precision_score(y_test, y_predRFC)*100
re_RFC = recall_score(y_test, y_predRFC)*100
f1_RFC = f1_score(y_test, y_predRFC)*100

""" k Nearest Neighbor classifier """
y_predkNN = classifierkNN.predict(X_test)
cm_kNN = confusion_matrix(y_test, y_predkNN)
ac_kNN = accuracy_score(y_test, y_predkNN)*100
pc_kNN = precision_score(y_test, y_predkNN)*100
re_kNN = recall_score(y_test, y_predkNN)*100
f1_kNN = f1_score(y_test, y_predkNN)*100

""" Logistic Regression classifier """
y_predLR = classifierLR.predict(X_test)
cm_LR = confusion_matrix(y_test, y_predLR)
ac_LR = accuracy_score(y_test, y_predLR)*100
pc_LR = precision_score(y_test, y_predLR)*100
re_LR = recall_score(y_test, y_predLR)*100
f1_LR = f1_score(y_test, y_predLR)*100

""" Naive Bayes Classifier """
y_predGaussNB = classifierGaussNB.predict(X_test)
cm_GaussNB = confusion_matrix(y_test, y_predGaussNB)
ac_GaussNB = accuracy_score(y_test, y_predGaussNB)*100
pc_GaussNB = precision_score(y_test, y_predGaussNB)*100
re_GaussNB = recall_score(y_test, y_predGaussNB)*100
f1_GaussNB = f1_score(y_test, y_predGaussNB)*100

""" Decision Tree Classifier """
y_predDTC = classifierDTC.predict(X_test)
cm_DTC = confusion_matrix(y_test, y_predDTC)
ac_DTC = accuracy_score(y_test, y_predDTC)*100
pc_DTC = precision_score(y_test, y_predDTC)*100
re_DTC = recall_score(y_test, y_predDTC)*100
f1_DTC = f1_score(y_test, y_predDTC)*100

""" Support Vector Machine """
y_predSVM = classifierSVM.predict(X_test)
cm_SVM = confusion_matrix(y_test, y_predSVM)
ac_SVM = accuracy_score(y_test, y_predSVM)*100
pc_SVM = precision_score(y_test, y_predSVM)*100
re_SVM = recall_score(y_test, y_predSVM)*100
f1_SVM = f1_score(y_test, y_predSVM)*100

from pandas import DataFrame
dict = {'Classifiers': ['Random Forest', 'k Nearest Neighbor', 'Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Support Vector Machine'],\
        'Accuracy Score': [ac_RFC, ac_kNN, ac_LR, ac_GaussNB, ac_DTC, ac_SVM],\
        'Precision Score': [pc_RFC, pc_kNN, pc_LR, pc_GaussNB, pc_DTC, pc_SVM],\
        'Recall Score': [re_RFC, re_kNN, re_LR, re_GaussNB, re_DTC, re_SVM],\
        'F1 Score': [f1_RFC, f1_kNN, f1_LR, f1_GaussNB, f1_DTC, f1_SVM]}
df = DataFrame(dict)
df.style