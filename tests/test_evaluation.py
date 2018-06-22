# -*- coding: utf-8 -*-
import pickle
import unittest
import numpy as np
import pandas as pd
from os import path
from scipy import sparse
from sklearn.metrics import mean_squared_error
# Original Models
from ridge.models import FacMac, CDFacMac


BASEDIR = path.dirname(path.abspath(__file__))
PATH2TEMP = path.join(BASEDIR, 'tmp')
PATH2FEATURES = path.join(PATH2TEMP, 'horseracing_sparse_features.npz')
PATH2TARGETS = path.join(PATH2TEMP, 'horseracing_targets.npy')
PATH2RACEIDS = path.join(PATH2TEMP, 'horseracing_raceids.pkl')
TRAIN_SIZE = 0.8
N_ITER = 20
N_ENTITIES = 5724
# N_TRAIN_ROWS = 30969


def rmse(obs, pred):
    """Calculate RMSE evaluation score.

    Parameters
    ----------
    obs   : np.ndarray, whose shape is (n_tests, ).
            An array of observed values.
    pred  : np.ndarray, whose shape is (n_tests, ).
            An array of predicted values.

    Return
    ------
    score : RMSE score.
    """
    assert(len(obs) == len(pred))
    return np.sqrt(np.sum(np.power(obs - pred, 2)) / len(obs))


class TestEvaluation(unittest.TestCase):

    def test_evaluation(self):
        # [START Loading data]
        features = sparse.csr_matrix(sparse.load_npz(PATH2FEATURES))
        targets = np.load(PATH2TARGETS)
        with open(PATH2RACEIDS, mode='rb') as fp:
            raceids = pd.Series(pickle.load(fp))  # whose dtype is `object'
        # [END Loading data]
        max_train_raceid = raceids.iloc[int(len(raceids) * TRAIN_SIZE)]
        n_train_rows = raceids[raceids == max_train_raceid].index[0]
        X_train = features[:n_train_rows]
        y_train = targets[:n_train_rows]
        X_test = features[n_train_rows:]
        y_test = targets[n_train_rows:]
        # [START Iteratively Fitting the model]
        for this_k in range(4, 5):
            # model = FacMac(type_X=sparse.csr_matrix).fit(X_train, y_train, k=this_k, n_iter=N_ITER, eta=1e-4)
            model = CDFacMac(type_X=sparse.csr_matrix).fit(X_train, y_train, N_ENTITIES, k=this_k, n_iter=N_ITER, eta=1e-4)
            y_pred = model.predict(X_test)
            score = mean_squared_error(y_test, y_pred)
            # [START Display Stats]
            print('Model Info')
            print(f'+ k: {model.k}')
            print(f'+ L2: {model.l2}')
            print(f'+ eta: {model.eta}')
            print('Evaluation')
            print(f'# of test: {len(y_test)}')
            print(f'RMSE Score: {np.round(score, 4)}')
            for idx in range(10):
                print(f'Race Ids: {raceids[n_train_rows+idx]}')
                print(f'Observed Samples: {y_test[idx]}')
                print(f'Prediction Samples: {y_pred[idx]}')
                print('---' * 20)
            # [END Display Stats]
            path2output = path.join(PATH2TEMP, f'pred-k{model.k}-L2{len(str(model.eta))-2}-iter{N_ITER}-CDFM.npy')
            np.save(path2output, y_pred)
        # [END Iteratively Fitting the model]


if __name__ == '__main__':
    unittest.main()
