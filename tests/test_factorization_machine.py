# -*- coding: utf-8 -*-
import unittest
import numpy as np
from tqdm import tqdm
from scipy import sparse
from ridge.models import FacMac


class TestFM(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
           #  Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
           # A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
            [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0 ],
            [1, 0, 0,  0,  1,  0,  0,   0.3, 0.3, 0.3, 0,     14,   1,  0,  0,  0 ],
            [1, 0, 0,  0,  0,  1,  0,   0.3, 0.3, 0.3, 0,     16,   0,  1,  0,  0 ],
            [0, 1, 0,  0,  0,  1,  0,   0,   0,   0.5, 0.5,   5,    0,  0,  0,  0 ],
            [0, 1, 0,  0,  0,  0,  1,   0,   0,   0.5, 0.5,   8,    0,  0,  1,  0 ],
            [0, 0, 1,  1,  0,  0,  0,   0.5, 0,   0.5, 0,     9,    0,  0,  0,  0 ],
            [0, 0, 1,  0,  0,  1,  0,   0.5, 0,   0.5, 0,     12,   1,  0,  0,  0 ],
        ])
        self.y = np.array([5, 3, 1, 4, 5, 1, 5])

    def test_fitting_fm_with_ndarray(self):
        X_train = self.X[0:5, :]
        y_train = self.y[0:5]
        X_test = self.X[5:, :]
        y_test = self.y[5:]

        print('Fitting FM with np.ndarray')
        model = FacMac().fit(X_train, y_train, k=4, n_iter=1000)
        print(f'pred: {model.predict(X_test)}')
        print(f'obs[0] : {y_test[0]}')
        print(f'obs[1] : {y_test[1]}')

    def test_fitting_fm_with_csr_matrix(self):
        sparse_X = sparse.csr_matrix(self.X)
        X_train = sparse_X[0:5, :]
        y_train = self.y[0:5, ]
        X_test = sparse_X[5:, :]
        y_test = self.y[5:]

        print('Fitting FM with sparse.csr_matrix')
        model = FacMac(type_X=sparse.csr_matrix).fit(X_train, y_train, k=4, n_iter=1000)
        print(f'type of X_train: {type(X_train)}')
        print(f'pred: {model.predict(X_test)}')
        print(f'obs[0] : {y_test[0]}')
        print(f'obs[1] : {y_test[1]}')


if __name__ == '__main__':
    unittest.main()
