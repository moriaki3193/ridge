# -*- coding: utf-8 -*-
import unittest
import numpy as np
from tqdm import tqdm
from ridge.models import FacMac


class TestFM(unittest.TestCase):

    def test_fm(self):
        # [START データセットの作成]
        X = np.array([
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
        y = np.array([5, 3, 1, 4, 5, 1, 5])
        X_train = X[0:5, :]
        y_train = y[0:5]
        X_test = X[5:, :]
        y_test = y[5:]
        # [END データセットの作成]

        model = FacMac().fit(X_train, y_train, k=4, n_iter=1000)
        print(f'pred: {model.predict(X_test)}')
        print(f'obs[0] : {y_test[0]}')
        print(f'obs[1] : {y_test[1]}')


if __name__ == '__main__':
    unittest.main()
