#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:08:17 2019

@author: Supriya, Nirav
"""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifierCV
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.utils import to_categorical
from keras import layers, optimizers
from keras.models import Sequential
from Code.DataOperations import DataOperations
import numpy as np
import warnings

warnings.filterwarnings("ignore")
class Model:
    
    def __init__(self, inputDataFrame):
        self.inputDataFrame = inputDataFrame
        
    def splitData(self):
        self.inputDataFrame['classification'] = DataOperations().debtTypeModifications(self.inputDataFrame)
        self.inputDataFrame['category_id'] = self.inputDataFrame['classification'].factorize()[0]
        
        X_train, X_test, y_train, y_test = train_test_split(self.inputDataFrame['commenttext'], 
                                                            self.inputDataFrame['category_id'], 
                                                            test_size=0.3, 
                                                            random_state = 100)
        return X_train, X_test, y_train, y_test
    
    def dataTransformNN(self):
        X_train, X_test, y_train, y_test = self.splitData()
        
        # transforming the data
        vectorizer = CountVectorizer()
        vectorizer.fit(X_train)
        
        X_train = vectorizer.transform(X_train)
        X_test = vectorizer.transform(X_test)
        
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        return X_train, X_test, y_train, y_test
    
    def results(self, y_test, y_preds, modelName):
        average = 'macro'
        accuracy = np.mean(y_test == y_preds)
        f1Score = f1_score(y_test, y_preds, average=average)
        precisionScore = precision_score(y_test, y_preds, average=average)
        recallScore = recall_score(y_test, y_preds, average=average)
        
        print("{0}: ".format(modelName))
        print("1. Accuracy: {:.3f} \n2. F1 Score: {:.4f} \n3. Precision: {:.4f} \n4. Recall: {:.4f}".format((accuracy*100), f1Score, precisionScore, recallScore))
        return accuracy, f1Score, precisionScore, recallScore
    
    def modelPipeline(self, algorithmSpecifications):
        modelPipeline = Pipeline([('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer()),
                                ('clf', algorithmSpecifications),
                               ])
        return modelPipeline
    
    def logisticRegression(self):

        X_train, X_test, y_train, y_test = self.splitData()
        logisticRegression = self.modelPipeline(LogisticRegression(n_jobs=20, C=1e5, solver='lbfgs', multi_class='multinomial'))
        logisticRegression.fit(X_train, y_train)
        y_preds = logisticRegression.predict(X_test)
        accuracy, f1Score, precisionScore, recallScore = self.results(y_test, y_preds, "Logistic Regression")
        return accuracy, f1Score, precisionScore, recallScore
    
    def svmSVC(self):
        X_train, X_test, y_train, y_test = self.splitData()
        svc = self.modelPipeline(svm.SVC(kernel='rbf'))
        svc.fit(X_train, y_train)
        y_preds = svc.predict(X_test)
        accuracy = np.mean(y_test == y_preds)
        accuracy, f1Score, precisionScore, recallScore = self.results(y_test, y_preds, "SVC")
        return accuracy, f1Score, precisionScore, recallScore
    
    def linearSVC(self):
        X_train, X_test, y_train, y_test = self.splitData()
        linearSVC = self.modelPipeline(LinearSVC(random_state=0, tol=1e-5))
        linearSVC.fit(X_train, y_train)
        y_preds = linearSVC.predict(X_test)
        accuracy = np.mean(y_test == y_preds)
        accuracy, f1Score, precisionScore, recallScore = self.results(y_test, y_preds, "Linear SVC")
        return accuracy, f1Score, precisionScore, recallScore
    
    def naiveBayes(self):
        X_train, X_test, y_train, y_test = self.splitData()
        naiveBayes = self.modelPipeline(MultinomialNB())        
        naiveBayes.fit(X_train, y_train)
        y_preds = naiveBayes.predict(X_test)
        accuracy = np.mean(y_test == y_preds)
        accuracy, f1Score, precisionScore, recallScore = self.results(y_test, y_preds, "Naive Bayes")
        return accuracy, f1Score, precisionScore, recallScore       

    def perceptron(self):
        X_train, X_test, y_train, y_test = self.splitData()
        perceptron = self.modelPipeline(Perceptron(tol=1e-3, random_state=0)) 
        perceptron.fit(X_train, y_train)
        y_preds = perceptron.predict(X_test)
        accuracy = np.mean(y_test == y_preds)
        accuracy, f1Score, precisionScore, recallScore = self.results(y_test, y_preds, "Perceptron: ")
        return accuracy, f1Score, precisionScore, recallScore
        
    def ridgeClassifierCV(self):
        X_train, X_test, y_train, y_test = self.splitData()
        ridgeClassifier = self.modelPipeline(RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]))
        ridgeClassifier.fit(X_train, y_train)
        y_preds = ridgeClassifier.predict(X_test)
        accuracy = np.mean(y_test == y_preds)
        accuracy, f1Score, precisionScore, recallScore = self.results(y_test, y_preds, "Ridge Classifier: ")
        return accuracy, f1Score, precisionScore, recallScore
        
    def neuralNetwork(self):
        X_train, X_test, y_train, y_test = self.dataTransformNN()
        
        input_dim = X_train.shape[1]
        
        model = Sequential()
        
        #input layer
        model.add(layers.Dense(128, input_dim=input_dim, activation='relu'))
        model.add(layers.Dropout(0.1))
        
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.1))

        # output layer
        model.add(layers.Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adadelta(), metrics=['accuracy'])
        test = model.fit(X_train, y_train, epochs=15,verbose=False, validation_data=(X_test, y_test), batch_size=10)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Neural Network Accuracy:  {:.3f} %".format(accuracy*100))
        return test, accuracy
