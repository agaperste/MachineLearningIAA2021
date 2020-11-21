#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:25:13 2020

@author: zhanyina

@citations: 
    https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e
    https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters
    https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
"""

import os, time
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import RobustScaler
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
from scikitplot.metrics import plot_lift_curve 
import plotly.graph_objs as go
import xgboost as xgb
from xgboost import plot_importance

sns.set_style("whitegrid")

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

#%% data exploration
target_bi1_train.describe()
train["binary.target1"].value_counts() #0: 217,448 | 1: 32,552 --> 86.9792% 0 
train.isnull().sum().sum() #No missing values

#%% create dummy variables
v45_dummied_train = pd.get_dummies(predictors_train["v45"], prefix="v45")
v45_dummied_valid = pd.get_dummies(predictors_valid["v45"], prefix="v45")
v45_dummied_test = pd.get_dummies(predictors_test["v45"], prefix="v45")

predictors_train = predictors_train.drop(columns=["v45"])
predictors_valid = predictors_valid.drop(columns=["v45"])
predictors_test = predictors_test.drop(columns=["v45"])

#%% Data pre-processing - Standardize data using median due to outliers / non-normally dist. variables

# scaler = RobustScaler()

# #Fit to training data
# scaler.fit(predictors_train)

# #Standardize Data (predictors) using mean and std from fit on X_train
# predictors_train = scaler.transform(predictors_train)
# predictors_valid = scaler.transform(predictors_valid)
# predictors_test = scaler.transform(predictors_test)

# no change on roc auc after standardizing data

#%% training XG Boost

# Predict binary target1
data_matrix = xgb.DMatrix(data=predictors_train, label=target_bi1_train)

# ''' grid search param tunning '''
# clf = xgb.XGBRegressor()
# parameters = {
#       "learning_rate"    : [0.051, 0.15], 
#       "max_depth"        : [3, 4], 
#       "colsample_bytree" : [0.7, 0.85],
#       "n_estimators"     : [15],
#         "reg_lambda"       : [0.8],
#       # "reg_alpha"       : [1e-5, 1e-2, 0.1, 1, 100]
#       }
# grid = GridSearchCV(clf, parameters, 
#                     n_jobs=2, # my Mac has 2 real cores
#                     scoring="roc_auc", # our objective to maximize auc
#                     cv=None)
# %time grid.fit(predictors_train, target_bi1_train)
# # 1h 8s

# # Print the best parameters found
# print(grid.best_params_)
# print(grid.best_score_)

xg_reg1 = xgb.XGBRegressor(objective = 'binary:logistic',
                            learning_rate = 0.051,
                            colsample_bytree = 0.85,
                            max_depth = 3,
                            n_estimators = 15, 
                            reg_lambda = 0.8
                            )
%time xg_reg1.fit(predictors_train, target_bi1_train)

%time y_pred = xg_reg1.predict(predictors_valid)

# #%% variable selection
# # plot feature importance
# plot_importance(xg_reg1)
# pyplot.show()
# print(xg_reg1.feature_importances_)

# # only keep vars that are important
# predictors_train = predictors_train[["v13", "v1", "v2", "v51", "v103", "v99", "v72", "v33", "v75", 
#                                      "v77", "v74", "v28"]]
# predictors_valid = predictors_valid[["v13", "v1", "v2", "v51", "v103", "v99", "v72", "v33", "v75", 
#                                      "v77", "v74", "v28"]]
# predictors_test = predictors_test[["v13", "v1", "v2", "v51", "v103", "v99", "v72", "v33", "v75", 
#                                      "v77", "v74", "v28"]]
#%% metrics
roc_auc = metrics.roc_auc_score(target_bi1_valid, y_pred)
print("roc_auc=", roc_auc)
# all variables --> 0.8462735117912732
# with selected variables --> 0.8459354449551453

y_pred = np.where(y_pred > 0.5, 1, 0)
cm = metrics.confusion_matrix(target_bi1_valid, y_pred)

precision = metrics.precision_score(target_bi1_valid, y_pred)
print("precision=", precision)

accuracy = metrics.accuracy_score(target_bi1_valid,y_pred)
print("accuracy=", accuracy)

#Plot Area Under Curve
plt.figure()
false_positive_rate, recall, thresholds = metrics.roc_curve(target_bi1_valid, 
                                                            y_pred)
roc_auc = metrics.auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()
print('AUC score:', roc_auc)

#Print Confusion Matrix
plt.figure()
labels = ['No Default', 'Default']
plt.figure(figsize=(8,6))
sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, 
            fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

# Lift
plot_lift_curve(target_bi1_valid, pd.get_dummies(y_pred).to_numpy())

#%% rolling up training and validation set
predictors_rolled = predictors_train.append(predictors_valid)
target_bi1_rolled = target_bi1_train.append(target_bi1_valid)

data_matrix2 = xgb.DMatrix(data=predictors_rolled, label=target_bi1_rolled)

xg_reg2 = xgb.XGBRegressor(objective = 'binary:logistic',
                            learning_rate = 0.051,
                            colsample_bytree = 0.85,
                            max_depth = 3,
                            n_estimators = 15,
                            reg_lambda = 0.8
                            )

%time xg_reg2.fit(predictors_rolled, target_bi1_rolled)

#%% predicting test set
target_bi1_test = xg_reg2.predict(predictors_test)
column_names = ["row", "binary.target1"]
result = pd.DataFrame(columns = column_names)
result.row = predictors_test.index
result["binary.target1"] = target_bi1_test

# export
result.to_csv("Blue5a.csv")

# sanity check
len(result) #85022
result_back = pd.read_csv("Blue5a.csv")
result_back.info() #85022 entries

