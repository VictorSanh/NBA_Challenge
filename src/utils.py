import numpy as np
from tqdm import tqdm
import pandas as pd

def split_train_val(data_X_train, data_Y_train, portion_train=0.7) :
    # Size of train
    nb_games = len(data_X_train)
    n_train = int(portion_train*nb_games)
    n_val = nb_games - n_train
    id_train = np.random.choice(nb_games, n_train, replace=False)

    # Remove ID
    X = data_X_train.as_matrix()[:,1:]
    Y = data_Y_train.as_matrix()[:,1:]

    # Def train and validation data
    X_train = X[id_train,:]
    Y_train = Y[id_train,:].reshape(n_train,)
    X_val = np.delete(X, id_train, axis = 0)
    Y_val = np.delete(Y, id_train, axis = 0).reshape(n_val,)

    return X_train, Y_train, X_val, Y_val


def extract_main_features(data_X_test):
    filter_col = [col for col in data_X_test if col.startswith('miss') or col.startswith('total rebound') or col.startswith('score') or col.startswith('assist')
    											or col.startswith('three points') or col.startswith('fied goals') or col.startswith('free throws')]
    data_X_test = data_X_test[filter_col]
    return data_X_test.as_matrix()


def feature_engineering(data):
	# Add Field Goals, Free throws, Three Points
	# Compute total rebound and total foul

	nb_checks = 1440
	nb_games = len(data)

	data['diff points_1'] = data['score_1']
	for i in range(1,nb_checks) :
		data['diff points_{}'.format(i+1)] = data['score_{}'.format(i+1)] - data['score_{}'.format(i)]


	# First second
	data['free throws_1'] =  np.ones(nb_games)*(data['diff points_1'] == 1) - np.ones(nb_games)*(data['diff points_1'] == -1)
	data['three points_1'] = np.ones(nb_games)*(data['diff points_1'] == 3) - np.ones(nb_games)*(data['diff points_1'] == -3)
	data['fied goals_1'] = data['score_1'] - data['free throws_1']
	data['total rebound_1'] = data['offensive rebound_1'] + data['defensive rebound_1']
	data['total foul_1'] = data['offensive foul_1'] + data['defensive foul_1']    

	# Other seconds
	for i in range(2,nb_checks+1) :
		data['free throws_{}'.format(i)] = np.add(data['free throws_{}'.format(i-1)].as_matrix(),
											np.ones(nb_games)*(data['diff points_{}'.format(i)] == 1) - np.ones(nb_games)*(data['diff points_{}'.format(i)] == -1)).astype(int)
		data['three points_{}'.format(i)] = np.add(data['three points_{}'.format(i-1)].as_matrix(),
											np.ones(nb_games)*(data['diff points_{}'.format(i)] == 3) - np.ones(nb_games)*(data['diff points_{}'.format(i)] == -3)).astype(int)
		# Field goals = points - free throws
		data['fied goals_{}'.format(i)] = data['score_{}'.format(i)] - data['free throws_{}'.format(i)]
		data['total rebound_{}'.format(i)] = data['offensive rebound_{}'.format(i)] + data['defensive rebound_{}'.format(i)]
		data['total foul_{}'.format(i)] = data['offensive foul_{}'.format(i)] + data['defensive foul_{}'.format(i)]

	return data
