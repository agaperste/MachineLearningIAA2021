#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:34:18 2020

@author: zhanyina

@citations: 
    https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
    https://stackoverflow.com/questions/41639557/how-to-perform-logistic-lasso-in-python
    https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_path.html
    https://towardsdatascience.com/andrew-ngs-machine-learning-course-in-python-regularized-logistic-regression-lasso-regression-721f311130fb
    https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/
    
"""
import os, time
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import statsmodels.api as sm
import seaborn as sns

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#%% importing the dataset
os.chdir("/Users/zhanyina/Documents/MSA/AA502 Analytics Methods and Applications I/Machine Learning/competition")
pd.set_option("display.max_columns", None)

train = pd.read_csv('MLProject21_train.csv')
train.info() #250,000 rows, 130 columns
train.head()

valid = pd.read_csv("MLProject21_valid.csv")
valid.info() #100,000 rows, 130 columns

test = pd.read_csv("MLProject21_test.csv")
test.info() #85,022 rows, 130 columns

#%% setting up predictors and targets
predictors_train = train.iloc[:, 0:126]
target_bi1_train = train['binary.target1']

predictors_valid = valid.iloc[:, 0:126]
target_bi1_valid = valid['binary.target1']

predictors_test = test.iloc[:, 0:126]


#%% create dummy variables
v45_dummied_train = pd.get_dummies(predictors_train["v45"], prefix="v45")
v45_dummied_valid = pd.get_dummies(predictors_valid["v45"], prefix="v45")
v45_dummied_test = pd.get_dummies(predictors_test["v45"], prefix="v45")

predictors_train = predictors_train.drop(columns=["v45"])
predictors_valid = predictors_valid.drop(columns=["v45"])
predictors_test = predictors_test.drop(columns=["v45"])


#%% recursive feature eliminination (RFE)
predictors_train_vars=predictors_train.columns.values.tolist()
y=['binary.target1']
X=[i for i in predictors_train_vars if i not in y]

logreg = LogisticRegression()
rfe = RFE(logreg, 20)
%time rfe = rfe.fit(predictors_train, target_bi1_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

#%% Implementing the model
X = predictors_train.loc[:, rfe.support_]
y=target_bi1_train
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

X = X.drop(columns=["v10"])

# do the same for validation
X_valid = predictors_valid.loc[:, rfe.support_]
X_valid = X_valid.drop(columns=["v10"])

# TODO do the same for test
#%% Logistic Regression Model Fitting

logreg = LogisticRegression(penalty='l1', solver='liblinear', max_iter=int(1e6),
                            warm_start=True,)
%time logreg.fit(X, y)

%time y_pred = logreg.predict(X_valid)

roc_auc = metrics.roc_auc_score(target_bi1_valid, y_pred)
print("roc_auc=", roc_auc) #0.5534, failed to converge

y_pred = np.where(y_pred > 0.5, 1, 0)
cm = metrics.confusion_matrix(target_bi1_valid, y_pred)

precision = metrics.precision_score(target_bi1_valid, y_pred)
print("precision=", precision)

accuracy = metrics.accuracy_score(target_bi1_valid,y_pred)
print("accuracy=", accuracy)

# getting confusion matrix
confusion_matrix = confusion_matrix(target_bi1_valid, y_pred)

plt.figure()
labels = ['0', '1']
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix, xticklabels = labels, yticklabels = labels, annot = True, 
            fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()
