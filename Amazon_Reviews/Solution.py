# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 22:24:55 2019

@author: phaneendra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("kindle_reviews.csv")

dataset.head()

dataset=dataset.loc[:,~dataset.columns.str.contains('^Unnamed')]


dataframe = dataset.sample(n=20000)

# takng columns whinh are helpful 
useful = ['helpful','overall', 'reviewText','summary']
dataframe = dataframe[useful]
dataframe[['Thumpsup','ThmpsDown']] = dataframe['helpful'].str.split(',',expand=True)

dataframe.Thumpsup = dataframe.Thumpsup.map(lambda x: str(x)[1:])
dataframe.ThmpsDown = dataframe.ThmpsDown.map(lambda x:str(x)[:-1])

del(dataframe['helpful'])

for i in list(dataframe):
    print(dataframe[i].isnull().value_counts())

# remove rows with null values

dataframe = dataframe[~dataframe['reviewText'].isnull()]
dataframe = dataframe[~dataframe['summary'].isnull()]

# assigning sentiment to reviews positive or negitive
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
senti = SentimentIntensityAnalyzer()
dataframe['Sentiment_text'] = dataframe['reviewText'].map(lambda x: senti.polarity_scores(x))


# =============================================================================
# 
# import nltk
# nltk.downloader.download('vader_lexicon')
# 
# =============================================================================

import nltk
nltk.download('wordnet')



