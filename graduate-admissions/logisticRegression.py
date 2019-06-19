# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 06:46:59 2019

@author: phaneendra
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os

dataset1 = pd.read_csv("Admission_Predict_Ver1.1.csv")

dataset1.info()

dataset1.columns

dataset1 = dataset1.rename(columns = {'LOR ':'LOR', 'Chance of Admit ':
    'Chance of Admit'})

dataframe1 = dataset1.drop('Serial No.',1)

y = dataframe1["Chance of Admit"]
x = dataframe1.drop(["Chance of Admit"], axis = 1)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

# Normalisation for getting more accurate results

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = scaler.fit_transform(X_test[X_train.columns])
y_train_1 = [1 if each>0.8 else 0 for each in y_train]
y_test_1  = [1 if each>0.8 else 0 for each in y_test]

y_train_1 = np.array(y_train_1)
y_test_1  = np.array(y_test_1)


# Logistic regerssion 
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,y_train_1)
predict_logistic = logistic_regression.predict(X_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test_1, predict_logistic)

from sklearn.metrics import precision_score

precision_score(y_test_1, predict_logistic)









