#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:08:17 2019

@author: Supriya
"""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import svm
import numpy as np

class Model:
    
    def __init__(self, inputDataFrame):
        self.inputDataFrame = inputDataFrame
        
    def splitData(self):
        self.inputDataFrame['category_id'] = self.inputDataFrame['classification'].factorize()[0]

        X_train, X_test, y_train, y_test = train_test_split(self.inputDataFrame['commenttext'], 
                                                            self.inputDataFrame['classification'], 
                                                            random_state = 0)
        return X_train, X_test, y_train, y_test
    
    def modelPipeline(self, algorithmSpecifications):
        modelPipeline = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', algorithmSpecifications),
               ])
        return modelPipeline
    
    def logisticRegression(self):
        X_train, X_test, y_train, y_test = self.splitData()
        logisticRegression = self.modelPipeline(LogisticRegression(n_jobs=10, 
                                                                   C=1e5, solver='lbfgs', multi_class='multinomial'))
        logisticRegression.fit(X_train, y_train)
        y_preds = logisticRegression.predict(X_test)
        print ("Logistic Regression Accuracy: ", np.mean(y_test == y_preds))
        
    def svmSVC(self):
        X_train, X_test, y_train, y_test = self.splitData()
        svc = self.modelPipeline(svm.SVC(kernel='linear'))
        svc.fit(X_train, y_train)
        y_preds = svc.predict(X_test)
        print ("SVM (SVC with Linear Kernal) Accuracy: ", np.mean(y_test == y_preds))
        
    def linearSVC(self):
        X_train, X_test, y_train, y_test = self.splitData()
        linearSVC = self.modelPipeline(LinearSVC(random_state=0, tol=1e-5))
        linearSVC.fit(X_train, y_train)
        y_preds = linearSVC.predict(X_test)
        print ("Linear SVC Accuracy: ", np.mean(y_test == y_preds))

'''
from DataOperations import DataOperations
inputFile = DataOperations().loadData("../Dataset/processedDataset.csv")

model = Model(inputFile)
model.logisticRegression()
model.linearSVC()
model.svmSVC()'''
