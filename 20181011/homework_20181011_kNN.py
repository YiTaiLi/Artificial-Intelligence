# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:17:35 2018

@author: Yi Tai
"""

from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

df = pd.read_csv('mlb_2017_regular_season_top_hitting.csv')
label_encoder = preprocessing.LabelEncoder()
df["Pos"] = label_encoder.fit_transform(df["Pos"]) #將字串數值化
#print(df.head(10))
df_data = df[["Pos","AVG?","OPS","OBP","RBI"]]
#print(df_data)
df_target = df["2017AllStar"]
train_X, test_X, train_y, test_y = train_test_split(df_data, df_target, test_size = 0.33)

#選擇K
Ks = np.arange(1, round(0.2 * len(train_X) + 1))
accuracies = []
for i in Ks:
    clf = neighbors.KNeighborsClassifier(n_neighbors = i)
    mlb_clf = clf.fit(train_X, train_y)
#    test_y_predicted = mlb_clf.predict(test_X)
    accuracy = mlb_clf.score(test_X, test_y)
    accuracies.append(accuracy)
plt.plot(Ks,accuracies)
plt.show()
Best_k_value=max(accuracies)
Index=accuracies.index(Best_k_value)+1
print("準確率:",clf.score(test_X,test_y))
print("最佳k值:",Index)
# 交叉驗證的k值最佳化
#Ks = np.arange(1, round(0.2 * len(train_X) + 1))
#accuracies = []
#for k in Ks:
#    clf = neighbors.KNeighborsClassifier(n_neighbors = k)
#    scores=cross_val_score(clf,train_X, train_y,scoring="accuracy",cv=10)
#    accuracies.append(scores.mean())
#plt.plot(Ks,accuracies)
#plt.show()




