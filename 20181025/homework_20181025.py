# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:33:38 2018

@author: Yi Tai
"""
"""
1.載入資料集
2.分類用SVM.SVC
3.參數的選擇
4.報告(正確率)
"""
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('mlb_2017_regular_season_top_hitting.csv' )
label_encoder = preprocessing.LabelEncoder()
df["Pos"] = label_encoder.fit_transform(df["Pos"])
df_data = df[["Pos","AVG?","OPS","OBP","SLG"]]
df_target = df["2017AllStar"]
X_train, X_test, y_train, y_test = train_test_split(
    df_data, df_target, test_size=0.3, random_state=0)
sc=StandardScaler()
train_X_std=sc.fit_transform(X_train)
test_X_std=sc.fit_transform(X_test)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3],
                     'C': [1,10, 100, 1000]}]

tuned_parameters2 = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000,1000]}]

clf = GridSearchCV(SVC(), tuned_parameters, cv=3)
clf.fit(X_train, y_train)

clf2 = GridSearchCV(SVC(), tuned_parameters2, cv=3)
clf2.fit(X_train, y_train)
print()
print("Kernel為rbf的參數報告")    
for params, mean_score, scores in clf.grid_scores_:
    clf= SVC(kernel=params['kernel'], C=params['C'], gamma=params['gamma'], probability=True)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predict)
    print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params),"accuracy: %.2f%%" % (100 * accuracy))
print()
print("Kernel為linear的參數報告")
for params, mean_score, scores in clf2.grid_scores_:
    clf2= SVC(kernel=params['kernel'], C=params['C'], probability=True)
    clf2.fit(X_train, y_train)
    predict = clf2.predict(X_test)
    accuracy2 = metrics.accuracy_score(y_test, predict)
    print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params),"accuracy: %.2f%%" % (100 * accuracy2))

