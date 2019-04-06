#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 23:03:15 2019

@author: Nirav, Supriya
"""
# load original dataset, summary regarding comments and deleted not required section which is here 'WITHOUT_CLASSIFICATION'
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from collections import Counter
import pandas as pd
import re

class DataOperations:

    def __init__(self):
        pass

    def loadData(self, fileName):
        dataset = pd.read_csv(fileName, index_col=None)
        dataset.dropna(inplace=True)
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
        
        print("Types of Debt and no of comments: ")
        for headers in range(0, len(columnHeaders)):
            for data in range(0, len(columnData)):
                if columnData[data] == columnHeaders[headers]:
                    count += 1
            print("{0}. {1} : {2}".format(headers+1,str(columnHeaders[headers]).lower(), count))
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
        regexForCodeList = ["(\n)", " //(.+?);", " //(.+?)\(\)", "{(.+?)}", "<(.+?)>", "\((.+?)\)", "@(.+?) "]
        specialChars = '[{0}]'.format("@#://().,$><%`~*-;\[\]=\"\'\{\}\|")
        stopWords = list(set(stopwords.words('english')))
        for i in range(0, len(tempList)):
            line = tempList[i].lower()
            for j in range(0, len(regexForCodeList)):
                line = re.sub(regexForCodeList[j], '', line)
            line = re.sub(specialChars, '', line)
            line = re.sub(r"\b[a-zA-Z]\b", "", line) # for removing single char
            line = line.replace("\\", "") # for removing slash
            
            # for removing stop words
            temp = list(line.split())
            temp = [word for word in temp if word not in stopWords]
            line = ' '.join(x for x in temp)
            
            # stemming
            #from nltk.stem import PorterStemmer
            #ps = PorterStemmer()
            #line = [ps.stem(word) for word in line.split()]
            #line = " ".join(str(x) for x in line)
            
            # updating dataFrame
            dataFrame.iloc[i, dataFrame.columns.get_loc('commenttext')] = line
        return dataFrame
    
    def certainDebtTypeWords(self, dataFrame, debtType):
        tempList = []
        tokenizer = TweetTokenizer()
        for elem in range(0, len(dataFrame)):
            if dataFrame['classification'][elem] == debtType:
                line = tokenizer.tokenize(dataFrame['commenttext'][elem])
                tempList.append(line)
        
        tempList = [item for sublist in tempList for item in sublist]
        wordCounter = Counter(tempList)
        mostCommonWords = wordCounter.most_common(200) # most common 200 words
        tempList = [elem[0] for elem in mostCommonWords]
        wordString = ' '.join(x for x in list(set(tempList)))
        return mostCommonWords, wordString
            