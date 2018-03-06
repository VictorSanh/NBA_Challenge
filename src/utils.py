import numpy as np

def preprocess(data_X_train, data_Y_train, portion_train=0.7) :
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
    filter_col = [col for col in data_X_test if col.startswith('miss') or col.startswith('offensive rebound') or col.startswith('score') or col.startswith('assist')]
    data_X_test = data_X_test[filter_col]
    return data_X_test.as_matrix()