import numpy as np
from tqdm import tqdm
import pandas as pd

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
    filter_col = [col for col in data_X_test if col.startswith('miss') or col.startswith('offensive rebound') or col.startswith('score') or col.startswith('assist')
    											or col.startswith('two pts') or col.startswith('three pts') or col.startswith('fg')]
    data_X_test = data_X_test[filter_col]
    return data_X_test.as_matrix()


def convert_pt(x, nb):
    """
    Convert score difference into 2/3 pts goals
    :param x: value, pandas columns
    :param nb: 2 or 3
    :return: differential value
    """
    if np.abs(x) == nb:
        return np.sign(x)
    else:
        return 0


def feature_engineering(df):
    # Add Two Points, Three Points and Field Goal
    nb_checks = 1440

    #Compute score difference
    df['diff score_%d' % 1] = df['score_%d' % 1]
    for k in tqdm(range(2,nb_checks+1)):
        df['diff score_%d' % k] = df['score_%d' % k] - df['score_%d' % (k-1)]

    #Compute 2pts/3pts
    for k in tqdm(range(1,nb_checks+1)):
        df['two pts_%d' % k] = df['diff score_%d' % k].apply(lambda x:convert_pt(x, 2))
        df['three pts_%d' % k] = df['diff score_%d' % k].apply(lambda x:convert_pt(x, 3))

    # Compute 2pts/3pts as cumulative sum
    two_pts_df = df[[k for k in df.columns if 'two pts' in k]].cumsum(axis=1)
    three_pts_df = df[[k for k in df.columns if 'three pts' in k]].cumsum(axis=1)
    df = pd.concat([df[[k for k in df.columns if 'two pts' not in k and 'three pts' not in k]],
                        two_pts_df,
                        three_pts_df], axis=1)

    # Compute FG and total rebound, total foul
    for k in tqdm(range(1,nb_checks+1)):
        df['fg_%d' % k] = df['two pts_%d' % k] + df['three pts_%d' % k]
        df['total rebound_%d' % k] = df['defensive rebound_%d' % k] + df['offensive rebound_%d' % k]
        df['total foul_%d' % k] = df['defensive foul_%d' % k] + df['offensive foul_%d' % k]

    keep = ['ID']
    return df[keep + sorted([k for k in df.columns if len(k.split('_')) > 1 and 'diff ' not in k],
                                       key=lambda x:int(x.split('_')[1]))]
