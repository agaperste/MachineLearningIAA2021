#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:06:13 2020

@author: zhanyina

@citations:
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    https://medium.com/edureka/support-vector-machine-in-python-539dca55c26a
"""
import os, time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

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

#%% preprocessing 

# # create dummy variables
# v45_dummied_train = pd.get_dummies(predictors_train["v45"], prefix="v45")
# v45_dummied_valid = pd.get_dummies(predictors_valid["v45"], prefix="v45")
# v45_dummied_test = pd.get_dummies(predictors_test["v45"], prefix="v45")

predictors_train = predictors_train.drop(columns=["v45"])
predictors_valid = predictors_valid.drop(columns=["v45"])
predictors_test = predictors_test.drop(columns=["v45"])

# # joining the dummied back
# predictors_train_2 = predictors_train.append(v45_dummied_train)
# predictors_valid_2 = predictors_valid.append(v45_dummied_valid)
# predictors_test_2 = predictors_test.append(v45_dummied_test)

# # Standardizing the features
# %time predictors_train_2 = StandardScaler().fit_transform(predictors_train_2)
# %time predictors_valid_2 = StandardScaler().fit_transform(predictors_valid_2)
# %time predictors_test_2 = StandardScaler().fit_transform(predictors_test_2)

# going to standardize non v45 variables, because null will create problems down the road and we know v45 is not important 
%time predictors_train = StandardScaler().fit_transform(predictors_train)
%time predictors_valid = StandardScaler().fit_transform(predictors_valid)
%time predictors_test = StandardScaler().fit_transform(predictors_test)

#%% PCA
num_pc = 10
'''
tried 50:
5 or 10 PCs seem to suffice
'''

pca = PCA(n_components=num_pc)
pc_cols = []
for i in range(1,num_pc+1):
    pc_cols.append("PC"+ str(i))
    
principalComponents_train = pca.fit_transform(predictors_train)
principalDf_train = pd.DataFrame(data = principalComponents_train
             , columns = pc_cols)

# Scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.savefig('scree_plot.png', dpi=300)
plt.show()

# applying the same to validation and test
principalComponents_valid = pca.fit_transform(predictors_valid)
principalDf_valid = pd.DataFrame(data = principalComponents_valid
             , columns = pc_cols)

principalComponents_test = pca.fit_transform(predictors_test)
principalDf_test = pd.DataFrame(data = principalComponents_test
             , columns = pc_cols)

#%% XGBoost
''' grid search param tunning '''
clf = xgb.XGBRegressor()
parameters = {
      "learning_rate"    : np.arange(0.1, 1.1, 0.1), 
      "max_depth"        : [3], 
      "colsample_bytree" : [0.85],
      "n_estimators"     : [15],
      "reg_lambda"       : [0.8],
      # "reg_alpha"       : [1e-5, 1e-2, 0.1, 1, 100]
      }
grid = GridSearchCV(clf, parameters, 
                    n_jobs=2, # my Mac has 2 real cores
                    scoring="roc_auc", # our objective to maximize auc
                    cv=None)
%time grid.fit(principalComponents_train, target_bi1_train)

# Print the best parameters found
print(grid.best_params_)
print(grid.best_score_)

'''training the model'''
xg_reg_pc = xgb.XGBRegressor(objective = 'binary:logistic',
                            learning_rate = 0.015,
                            colsample_bytree = 0.85,
                            max_depth = 3,
                            n_estimators = 35, 
                            reg_lambda = 0.85
                            )
%time xg_reg_pc.fit(principalComponents_train, target_bi1_train)

%time y_pred = xg_reg_pc.predict(principalComponents_valid)

roc_auc = metrics.roc_auc_score(target_bi1_valid, y_pred)
print("roc_auc=", roc_auc)
# 0.7289967756742065

# getting confusion matrix
y_pred = np.where(y_pred > 0.5, 1, 0)
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

#%% Logistic Regression Model Fitting
logreg = LogisticRegression(penalty='l2', solver='liblinear')
%time logreg.fit(principalComponents_train, target_bi1_train)

%time y_pred = logreg.predict(principalComponents_valid)

roc_auc = metrics.roc_auc_score(target_bi1_valid, y_pred)
print("roc_auc=", roc_auc)
# 0.5400472000591113

#%% Neural Network: Multi-Layer Perceptron Classifier Model
mlp = MLPClassifier(hidden_layer_sizes=(100,50,3),activation='logistic', max_iter=500)

#Fit training data to MLP model
%time mlp.fit(principalComponents_train, target_bi1_train) 
# 17 min

#Predictions and Evaluation
y_pred = mlp.predict(principalComponents_valid)

roc_auc = metrics.roc_auc_score(target_bi1_valid, y_pred)
print("roc_auc=", roc_auc)
# 0.5724498476693878

#%% SVM 
#create a classifier
cls = svm.SVC(kernel="linear")
#train the model
%time cls.fit(principalComponents_train, target_bi1_train)
#predict the response
%time y_pred = cls.predict(principalComponents_valid)

roc_auc = metrics.roc_auc_score(target_bi1_valid, y_pred)
print("roc_auc=", roc_auc)