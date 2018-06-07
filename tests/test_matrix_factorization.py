# -*- coding: utf-8 -*-
import dill
import unittest
import numpy as np
import pandas as pd
from os import path
from tqdm import tqdm
from ridge.models import MatFac


PATH2FEATURES = path.join(path.dirname(path.abspath(__file__)), 'data', 'u.data')
PATH2OUTPUT = path.join(path.dirname(path.abspath(__file__)), 'tmp', 'test_mf_model.pkl')


class TestMatFac(unittest.TestCase):

    def test_matrix_factorization(self):
        names = ('USERID', 'ITEMID', 'RATING', 'TIMESTAMP')
        dat = pd.read_csv(PATH2FEATURES, delimiter='\t', header=None, names=names)
        n_users = len(dat['USERID'].unique())
        n_items = len(dat['ITEMID'].unique())
        ratings = np.zeros((n_users, n_items))

        # [START レーティング行列の作成]
        for user_id, user_data in tqdm(dat.groupby('USERID')):
            user_index = user_id - 1  # 0-indexのため
            for _, row in user_data.iterrows():
                item_index = row.loc['ITEMID'] - 1  # 0-indexのため
                item_rating = row.loc['RATING']
                ratings[user_index, item_index] = item_rating
        # [END レーティング行列の作成]

        # [START レーティング行列の作成のテスト]
        sample_user_index = 123
        sample_user_id = sample_user_index + 1
        for _, row in dat[dat['USERID']==sample_user_id].iterrows():
            sample_item_id = row.loc['ITEMID']
            sample_item_index = sample_item_id - 1
            dat_rating = row.loc['RATING']
            mat_rating = ratings[sample_user_index, sample_item_index]
            self.assertEqual(dat_rating, mat_rating)
        # [END レーティング行列の作成のテスト]

        # [START Matrix Factoriztion モデルの学習]
        mf = MatFac()
        mf.fit(ratings, k=5, n_iter=1000, alpha=0.0002, verbose=True)
        # [END Matrix Factoriztion モデルの学習]

        # [START モデルの書き出し]
        with open(PATH2OUTPUT, 'wb') as fp:
            dill.dump(mf, fp)
        # [END モデルの書き出し]


if __name__ == '__main__':
    unittest.main()
