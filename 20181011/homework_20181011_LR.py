# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:17:35 2018

@author: Yi Tai
"""
from sklearn.metrics import confusion_matrix
from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Source
from sklearn import tree
from sklearn import preprocessing, linear_model
from sklearn.feature_selection import f_regression
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing  import StandardScaler

df = pd.read_csv('mlb_2017_regular_season_top_hitting.csv')
label_encoder = preprocessing.LabelEncoder()
df["Pos"] = label_encoder.fit_transform(df["Pos"]) #將字串數值化
print(df.head(10))
df_data = df[["Pos","AVG?","OPS","OBP","SLG","RBI"]]
print(df_data)
df_target = df["2017AllStar"]
train_X, test_X, train_y, test_y = train_test_split(df_data, df_target, test_size = 0.3)

logistic_regr = linear_model.LogisticRegression()
logistic_regr.fit(df_data, df_target)
print("係數:",logistic_regr.coef_)
print("截距:",logistic_regr.intercept_ )
print(f_regression(df_data, df_target)[1])

survived_predictions = logistic_regr.predict(df_data)
accuracy = logistic_regr.score(df_data, df_target)
print("準確率:",accuracy)

#邏輯回歸
#plt.style.use('ggplot')
#plt.rcParams['font.family']='SimHei'
#
#df=pd.read_csv("mlb_2017_regular_season_top_hitting.csv")
#
#x_train,x_test,y_train,y_test=train_test_split(df_data,df_target,test_size=0.3,random_state=20170816)
#
#sc=StandardScaler()
#sc.fit(x_train)
#
#x_train_nor=sc.transform(x_train)
#x_test_nor=sc.transform(x_test)
#
#lr=LogisticRegression()
#lr.fit(x_train_nor,y_train)
#
#lr=LogisticRegression()
#lr.fit(x_train_nor,y_train)
#
#print("係數:",lr.coef_)
#print("截距:",lr.intercept_ )
#np.round(lr.predict_proba(x_test_nor),3)
#
#cnf=confusion_matrix(y_test,lr.predict(x_test_nor))
#print('混淆矩陣: ',cnf)




