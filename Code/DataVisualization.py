#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:54:31 2019

@author: Nirav
"""
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

class DataVisualization:
    
    def __init__(self):
        pass

    def verticalBarGrapgh(self, dataFrame, columnName):
        plt.figure(figsize=(8,6))
        dataFrame.groupby(columnName).commenttext.count().plot.bar(ylim=0)
        plt.show()
        
    def horizontalBarGrapgh(self, x, y):
        fig, ax = plt.subplots()    
        width = 0.75  
        ind = np.arange(len(y))
        ax.barh(ind, y, width)
        ax.set_yticks(ind + width/2)
        ax.set_yticklabels(x, minor=False)
        plt.title('Accuracy Comparison')
        plt.xlabel('model name')
        plt.ylabel('model accuracy') 
        plt.axvline(x=11,linewidth=1, color='k')
        for i, v in enumerate(y):
            ax.text(v + 0.5, i + .125, str(v), fontweight='bold')   
        plt.show()
        
    def lineGraph(self, firstValue, secondValue):
        plt.plot(firstValue, marker='x', linestyle='-', color='b', label='Training Accuracy')
        plt.plot(secondValue, marker='o', linestyle='--', color='r', label='Validation Accuracy')
        plt.xlabel('No of Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training vs Validation Accuracy')
        plt.legend()
        plt.show()
        
    def wordCloud(self, wordString):
        wordcloud = WordCloud(max_font_size=30, background_color='white').generate(wordString)
        plt.figure(figsize=(13, 13))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
        