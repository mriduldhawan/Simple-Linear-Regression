# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 23:54:46 2020

@author: Mridul
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('salary_Data.csv')
# Splitting into X and Y
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# Splitting dataset imto train and test

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30, random_state=0)

slr = LinearRegression()
slr.fit(X_train,y_train)
y_predict = slr.predict(X_test)

#Implement Graph
plt.scatter(X_train,y_train,color='blue')
plt.plot(X_train,slr.predict(X_train))
plt.show()