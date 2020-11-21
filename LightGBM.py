#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 20:37:28 2020

@author: Jackie

@citations:
    https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc
    https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    https://lightgbm.readthedocs.io/en/latest/Parameters.html
    https://neptune.ai/blog/lightgbm-parameters-guide
    https://www.kaggle.com/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761
    https://www.kaggle.com/georsara1/light-gbm-solution-for-credit-fraud-detection
    https://www.kaggle.com/vincentlugat/lightgbm-plotly

"""
import os
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from scikitplot.metrics import plot_lift_curve 
import lightgbm as lgb

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
target_bi1_train = train.iloc[:, 128]

predictors_valid = valid.iloc[:, 0:126]
target_bi1_valid = valid.iloc[:, 128]

predictors_test = test.iloc[:, 0:126]

#%% data exploration
target_bi1_train.describe()
train["binary.target1"].value_counts() #0: 217,448 | 1: 32,552 --> 86.9792% 0 
train.isnull().sum().sum() #No missing values

#%% one hot encoding categorical variables
v45_dummied_train = pd.get_dummies(predictors_train["v45"], prefix="v45")
v45_dummied_valid = pd.get_dummies(predictors_valid["v45"], prefix="v45")
v45_dummied_test = pd.get_dummies(predictors_test["v45"], prefix="v45")

predictors_train = predictors_train.drop(columns=["v45"])
predictors_valid = predictors_valid.drop(columns=["v45"])
predictors_test = predictors_test.drop(columns=["v45"])

#%% Feature Scaling
sc = StandardScaler()
predictors_train = sc.fit_transform(predictors_train)
predictors_valid = sc.transform(predictors_valid)
predictors_test = sc.transform(predictors_test)

#%% training light GBM -- manual grid search --> 

def grid_search(lambda_l2):
    d_train = lgb.Dataset(predictors_train, label=target_bi1_train) 
    params = {}
    
    '''tried 
        0.003->0, 0.006->0, 0.011->0.4, ***0.016->0.712***, 0.021->0.633, 0.026->0.605,
        0.031->0.548, 0.036-> 0.505, 0.041->0.497, 0.046->0.497, 0.051->0.492,
        0.056->0.5, 0.0361->0.500...
        
        0.015->0.6902, 0.016->0.712, 0.017->0.671, 0.018->0.684, 0.019->0.653, 0.020->0.665
        
    '''
    params['learning_rate'] = 0.016
    # params['lambda_l1'] = lambda_l1 #0.8
    params['lambda_l2'] = 0.9 #0.9
    
    '''tried: gbdt, goss'''
    params['boosting_type'] = 'goss'
    params['objective'] = 'binary'
    params['metric'] = 'auc'
    params['feature_fraction'] = 0.8
    
    #tried 10, 20, 30, 40, 50, 60, 60, 80, 90, 100, 500, all ppv=0
    # params['num_leaves'] = 100 
    
    # params['min_data'] = 50
    
    '''tried 
        ***4->0.881***, 5->0.742, 
        6->0.702, 7->0.675, 8->0.679, 9->0.723...11->0.728
    '''
    # params['max_depth'] = max_depth
    
    %time clf = lgb.train(params, d_train, 100)
    
    y_pred=clf.predict(predictors_valid)

    #Confusion matrix
    cm = confusion_matrix(target_bi1_valid, y_pred)
    
    return cm

for num in np.arange(0.1, 0.9, 0.1):
    result = grid_search(0.9)
    
fpr1,tpr1,thresh1 = roc_curve(target_bi1_valid, y_pred)
roc_auc1 = auc(fpr1, tpr1)
roc_auc1 #0.8441337301472112

#%% training light GBM -- automatic grid search
d_train = lgb.Dataset(predictors_train, label=target_bi1_train)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'learning_rate': 0.016,
          'colsample_bytree': 0.85,
          "n_estimators": 15
          }

# Create parameters to search
gridParams = {
    'boosting_type' : ['gbdt', "goss"],
    'learning_rate': [0.016, 0.15],
    'n_estimators': [5, 10, 15],
    'colsample_bytree' : [0.65, 0.85],
    }

# Create classifier to use
mdl = lgb.LGBMClassifier(
          objective = 'binary',
          num_threads = 2, 
          random_state = 720,
          metric = 'auc',
          silent = True,
          
          boosting_type= params['boosting_type'],
          learning_rate = params["learning_rate"],
          colsample_bytree = params["colsample_bytree"],
          n_estimators = params["n_estimators"]
          )

# View the default model params:
mdl.get_params().keys()

# Create the grid
grid = GridSearchCV(mdl, gridParams, verbose=2, cv=None, n_jobs=2)

# Run the grid
grid.fit(predictors_train, target_bi1_train)

# Print the best parameters found
print(grid.best_params_)
print(grid.best_score_)

# Using parameters already set above, replace in the best from the grid search
params['n_estimators'] = grid.best_params_['n_estimators']
params['learning_rate'] = grid.best_params_['learning_rate']
params['colsample_bytree'] = grid.best_params_['colsample_bytree']

print('Fitting with params: ')
print(params)

#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                  d_train,
                  280,
                  #early_stopping_rounds= 40,
                  verbose_eval= 4
                  )

# Predict on test set
predictions_lgbm_prob = lgbm.predict(predictors_valid)
predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output

predictions_lgbm_01 = np.where(y_pred > 0.5, 1, 0) #Turn probability to 0-1 binary output
np.average(y_pred)

#%% Print accuracy measures and variable importances

#Plot Variable Importances
lgb.plot_importance(lgbm, max_num_features=21, importance_type='split')

#Print accuracy
acc_lgbm = accuracy_score(target_bi1_valid,predictions_lgbm_01)
print('Overall accuracy of Light GBM model:', acc_lgbm)

#Print Area Under Curve
plt.figure()
false_positive_rate, recall, thresholds = roc_curve(target_bi1_valid, predictions_lgbm_prob)
roc_auc = auc(false_positive_rate, recall)
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
# cm = confusion_matrix(target_bi1_valid, predictions_lgbm_01)
labels = ['0', '1']
plt.figure(figsize=(8,6))
sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.savefig('CM_LightGBM.png', dpi=300)
plt.show()

# Lift
plot_lift_curve(target_bi1_valid, pd.get_dummies(y_pred).to_numpy())