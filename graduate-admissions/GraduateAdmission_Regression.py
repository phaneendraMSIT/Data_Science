# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 10:07:55 2019

@author: phaneendra
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os
import FIleScore as fls
dataset1 = pd.read_csv("Admission_Predict_Ver1.1.csv")

dataset1 = dataset1.rename(columns = {'LOR ':'LOR', 'Chance of Admit ':
    'Chance of Admit'})

dataframe1 = dataset1.drop('Serial No.',1)
y = dataframe1["Chance of Admit"]
x = dataframe1.drop(["Chance of Admit"], axis = 1)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

X_testCopy = X_test.copy()

# finding corelation amoung the given variables
fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(dataframe1.corr(), ax=ax, annot=True, linewidths=0.05, 
            fmt= '.2f',cmap="magma")
plt.show()

# correlation between University rating and CGPA 

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

# Normalisation for getting more accurate results

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = scaler.fit_transform(X_test[X_train.columns])

# Linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

predictor_linear = regressor.predict(X_test)

from sklearn import metrics
linear = metrics.r2_score(y_test,predictor_linear)

# Gradient Boosting Regression 
from sklearn.ensemble import GradientBoostingRegressor
GBRegressor = GradientBoostingRegressor(n_estimators=500, max_depth=4, 
                                        min_samples_split=2, learning_rate=0.01,
                                        loss='ls')
GBRegressor.fit(X_train, y_train)
predictor_GBRegerssor = GBRegressor.predict(X_test)
GradientBoost = metrics.r2_score(y_test,predictor_GBRegerssor)

# RandomForest Regression 

from sklearn.ensemble import RandomForestRegressor

RFRegressor = RandomForestRegressor(max_depth=4, 
                                    random_state=0, n_estimators=100)

RFRegressor.fit(X_train,y_train)
predictor_RFRegressor = RFRegressor.predict(X_test)
RandomForest = metrics.r2_score(y_test, predictor_RFRegressor)

# KNN Regression 

# to find best value for neighbors 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

params = {'n_neighbors': [2,3,7,8]}
knn = KNeighborsRegressor()
model = GridSearchCV(knn,params,cv=5)
model.fit(X_train,y_train)
model.best_params_

# taking neighbours values as 7 is best

knn_regressor = KNeighborsRegressor(n_neighbors = 7)
knn_regressor.fit(X_train,y_train)
predict_knn = knn_regressor.predict(X_test)
KNN_Score = metrics.r2_score(y_test,predict_knn)

# Comparison of Different Regression Methods

Y  = np.array( [ GradientBoost, RandomForest, linear, KNN_Score])
X = [ "GredientBoost", "RandomForest", "Linear", "KNN Regression"]

plt.bar(X,Y)
plt.title("Comparision of Regression")
plt.xlabel("regression")
plt.ylabel("R2 Score")
plt.show()



# Creating a test file 
gre  = input("GRE:")
toefl = input("TOEFL: ")
UniRating = input("University Rating: ")
sop = fls.dale_chall_readability_score()
lor = fls.dale_chall_readability_score()
gpa = input("GPA: ")
research = input("research: ")
data_array = np.array([gre,toefl,UniRating,sop,lor,gpa,research])
newindex = X_testCopy.index[-1]+1
dataset_Test = X_testCopy.append(pd.DataFrame(index = [newindex], data = [data_array],
                               columns = X_testCopy.columns))

dataset_Test[dataset_Test.columns] = scaler.fit_transform(dataset_Test[dataset_Test.columns])
# dataset_Test.iloc[[-1],:]

predictor_linear_1 = regressor.predict(dataset_Test.iloc[[-1],:])

predict_Gradient_1 = GBRegressor.predict(dataset_Test.iloc[[-1],:])

predict_RandomForest_1 = RFRegressor.predict(dataset_Test.iloc[[-1],:])

predict_KNN_1 = knn_regressor.predict(dataset_Test.iloc[[-1],:])






