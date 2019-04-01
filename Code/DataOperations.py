#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 23:03:15 2019

@author: Nirav, Supriya
"""
# load original dataset, summary regarding comments and deleted not required section which is here 'WITHOUT_CLASSIFICATION'
import pandas as pd
import re

class DataOperations:

    def __init__(self):
        pass

    def loadData(self, fileName):
        dataset = pd.read_csv(fileName, index_col=None)
        dataset.reset_index(drop=True, inplace=True)
        return dataset

    def writeCSV(self, dataFrame, fileName):
        fileName = fileName + '.csv'
        dataFrame.to_csv(fileName, sep=',', encoding='utf-8', index=None)
        print("Writing file finished")

    def debtTypeStats(self, dataFrame, columnName):
        columnHeaders = list(set(dataFrame[columnName]))
        columnData = list(dataFrame[columnName])
        count = 0

        for headers in range(0, len(columnHeaders)):
            for data in range(0, len(columnData)):
                if columnData[data] == columnHeaders[headers]:
                    count += 1
            print(columnHeaders[headers], count)
            count = 0

    def filterDataFrame(self, dataFrame, debtType):

        processedDataset = dataFrame.copy(deep=True)
        tempList = []
        columnData = list(dataFrame['classification'])
        for data in range(0, len(columnData)):
            if columnData[data] == debtType:
                tempList.append(data)

        processedDataset.drop(processedDataset.index[tempList], inplace=True)
        processedDataset.drop('projectname', axis=1, inplace=True)
        processedDataset.reset_index(drop=True, inplace=True)
        return processedDataset
    
    def debtTypeModifications(self, dataFrame):
        otherDebtType = ['DOCUMENTATION', 'TEST']
        requirementDebtType = ['DEFECT', 'IMPLEMENTATION']
        
        for i in range(0, len(otherDebtType)):
            dataFrame.loc[dataFrame['classification'] == otherDebtType[i], 'classification'] = 'OTHERS'
            dataFrame.loc[dataFrame['classification'] == requirementDebtType[i], 'classification'] = 'REQUIREMENT'
            
        return dataFrame
    
    def removeChracters(self, dataFrame):
        tempList = list(dataFrame['commenttext'])
        specialChars = '[{0}]'.format('@#://().",*-;\[\]')
        for i in range(0, len(tempList)):
            line = re.sub(specialChars, '', tempList[i].lower())
            # html tags removal
            #line = re.sub('<[^>]*>', '', line)
            
            # stemming
            #from nltk.stem import PorterStemmer
            #ps = PorterStemmer()
            #line = [ps.stem(word) for word in line.split()]
            #line = " ".join(str(x) for x in line)
            dataFrame.iloc[i, dataFrame.columns.get_loc('commenttext')] = line
            

'''
dataOperations = DataOperations()
inputFile = dataOperations.loadData("../Dataset/technical_debt_dataset.csv")

dataOperations.debtTypeStats(inputFile, "classification")

processedDataset = dataOperations.filterDataFrame(inputFile, "WITHOUT_CLASSIFICATION")

dataOperations.writeCSV(processedDataset, "../Dataset/processedDataset")'''

