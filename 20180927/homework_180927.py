# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 12:42:52 2018

@author: Yi Tai
"""

import pandas as pd
from graphviz import Source
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
#import pandas_datareader as web

# load data
df = pd.read_csv('mlb_2017_regular_season_top_hitting.csv')
df = df[['OBP', 'AVG?', 'OPS', 'AB','2017Si.S']]

## preprocessing string value
#label_encoder = preprocessing.LabelEncoder()
#df['Team'] = label_encoder.fit_transform(df['Team'])

x = df[['OBP','AVG?','OPS','AB']]
y = df['2017Si.S']

# split train & test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3) 

# standard scaler
sc = preprocessing.StandardScaler()
sc.fit(x)

x_train_nor = sc.transform(x_train)
x_test_nor = sc.transform(x_test)

# create decision tree
decision_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth = 3)
decision_tree_clf = decision_tree.fit(x_train_nor, y_train)

# show the decision tree by graphic (need install graphviz package)
decision_tree_dot = tree.export_graphviz(decision_tree_clf, out_file = None, feature_names = x.columns)
decision_tree_source = Source(decision_tree_dot, filename = 'decision_tree')
decision_tree_source.view()

# create random forest
random_forest = ensemble.RandomForestClassifier(n_estimators = 100)
random_forest_clf = random_forest.fit(x_train_nor, y_train)

# predicted
y_tree_test_predicted = decision_tree_clf.predict(x_test_nor)
y_forest_test = random_forest_clf.predict(x_test_nor)

# show the score
tree_accuracy = metrics.accuracy_score(y_test, y_tree_test_predicted)
forest_accuracy = metrics.accuracy_score(y_test, y_forest_test)
print('tree_accuracy:', tree_accuracy)
print('forest_accuracy:', forest_accuracy)
