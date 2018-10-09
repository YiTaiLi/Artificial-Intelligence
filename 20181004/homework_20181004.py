# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:36:40 2018

@author: Yi Tai
"""

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier 
from sklearn import cross_validation, ensemble, preprocessing, metrics
from sklearn.model_selection import GridSearchCV

df= pd.read_csv('mlb_2017_regular_season_top_hitting.csv')

df_X = pd.DataFrame([df["AVG?"],df["OBP"],df["OPS"]]).T
df_y = df["2017AllStar"]
train_X, test_X, train_y, test_y = cross_validation.train_test_split(df_X, df_y, test_size = 0.3)

#AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)
bdt = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=20,min_samples_leaf=5),algorithm='SAMME.R')
#print(bdt)
param_test1 = {'n_estimators': range(50,300,50),"learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
gsearch1 = GridSearchCV(bdt,param_test1,cv=10)
gsearch1.fit(df_X,df_y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
means = gsearch1.cv_results_['mean_test_score']
stds = gsearch1.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gsearch1.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
    print()
print(gsearch1.best_params_, gsearch1.best_score_)

#test_y_predicted = boost.predict(test_X)
#accuracy = metrics.accuracy_score(test_y, test_y_predicted)
#print(accuracy)

#clf = AdaBoostClassifier(n_estimators=100)
#scores = cross_val_score(clf, df[['AVG?','OBP','SLG','OPS']], df["2017AllStar"])
#print(scores.mean())