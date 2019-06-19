# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 22:51:11 2019

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

dataset1 = dataset1.rename(columns = {'LOR ':'LOR'})

dataframe1 = dataset1.drop('Serial No.',1)

# from the plotting it is shown GRE, TOEFL, CGPA are more important to 
# calculate chance of admission. 
fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(dataframe1.corr(), ax=ax, annot=True, linewidths=0.05, 
            fmt= '.2f',cmap="magma")
plt.show()

# correlation between CGPA and University Ratings

plt.scatter(dataset1["University Rating"],dataset1.CGPA)
plt.title("CGPA Scores and University Ratings")
plt.xlabel("University Rating")
plt.ylabel("CGPA")
plt.show() 
# it is shown higher CGPA correlated with high University rating

# Correlation between CGPA and GRE Score. 

plt.scatter(dataset1["GRE Score"], dataset1["CGPA"])
plt.title("CGPA And GRE Score")
plt.xlabel("GRE Score")
plt.ylabel("CGPA")
plt.show()
# it is shown higher CGPA correlated with higer GRE Score

# K-NN Algorithm Classifer 

dataframe1 = dataframe1.rename(columns = {'Chance of Admit ':
    'Chance of Admit'} )

y = dataframe1["Chance of Admit"]
x = dataframe1.drop(["Chance of Admit"], axis = 1)

# Preparing Train and Tset data  
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
# to select best best k value(center)
list_center = [3, 4, 7, 8]

from sklearn.neighbors import KNeighborsClassifier 
scores = []
for i in list_center:
    knn_n = KNeighborsClassifier(n_neighbors=i)
    knn_n.fit(X_train,y_train_1)
    scores.append(knn_n.score(X_test,y_test_1))

# k = 3 gives highest score

knn_main = KNeighborsClassifier(n_neighbors=3)
knn_main.fit(X_train, y_train_1)
prediction = knn_main.predict(X_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test_1, prediction)

#KNN Regression 

from sklearn.neighbors import KNeighborsRegressor
knn_regressor = KNeighborsRegressor(n_neighbors=3)
knn_regressor.fit(X_train,y_train)
predict_regressor = knn_regressor.predict(X_test)









