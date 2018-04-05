#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
First best submission: Feature Engineering + XGBoost
'''

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from src.utils import *


# Load raw data
Data_X_train = pd.read_csv('data/challenge_fichier_dentrees_dentrainement_challenge_nba/train.csv')
Data_Y_train = pd.read_csv('data/challenge_fichier_de_sortie_dentrainement_challenge_nba.csv', sep=';')
Data_X_test = pd.read_csv('data/challenge_fichiers_dentrees_de_test_challenge_nba/test.csv')
print('Data Loaded')


# Feature Engineering
Data_X_train = feature_engineering(Data_X_train, two_points = True)
Data_X_test = feature_engineering(Data_X_test, two_points = True)

nb_games_test, col = Data_X_test.shape
nb_games_train, _ = Data_X_train.shape
nb_features = int((col-1)/1440)

X_train = Data_X_train.as_matrix()[:,1:]
score_end_train = X_train[:,25903]
X_train_models = X_train.reshape((nb_games_train, nb_features, 10, -1), order = 'F')
X_train_models = X_train_models.mean(axis = 2)
X_train_models = X_train_models.reshape(nb_games_train, nb_features*144)
X_train_tot = np.hstack((X_train_models, score_end_train))

X_test = Data_X_test.as_matrix()[:,1:]
score_end_test = X_test[:,25903]
X_test_models = X_test.reshape((nb_games_test, nb_features, 10, -1), order = 'F')
X_test_models = X_test_models.mean(axis = 2)
X_test_models = X_test_models.reshape(nb_games_test, nb_features*144)
X_test_tot = np.hstack((X_test_models, score_end_test))
print('Feature engineered')


# Train model
xgb = GradientBoostingClassifier(max_depth=10, n_estimators = 1000)
xgb.fit(X_train_tot, Y_train.ravel())
print('Model trained')


# Predict test observations
Y_pred_xgb = xgb.predict(X_test_tot)

ID_test = Data_X_test.iloc[:,0].as_matrix()
d = {'ID': ID_test, 'label': Y_pred_xgb}
Results_test = pd.DataFrame(data=d)
print('Test set predicted')


# Write predictions
Results_test.to_csv('predictions/XGBoost.csv', sep=';', index=False)