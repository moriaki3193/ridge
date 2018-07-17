# -*- coding: utf-8 -*-
import unittest
import numpy as np
import pandas as pd
from . import DATADIR
from os.path import join
from ridge.models import ConditionalLogit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


"""
Horse Racing Dataset

JRAにより2010年から2016年にかけて開催されたレースのうち
500万下条件以上のクラスのレースデータを集計したもの．

Stats
-----
+ The number of races: 12924
+ Avg number of horse entries: 14.25(±2.61)
+ The number of unique horses: 17859
+ Avg number of total runs: 10.31(±9.37)
"""


class TestConditionalLogit(unittest.TestCase):

    def test_conditional_logit(self):
        df = pd.read_csv(join(DATADIR, 'horseracing.csv'))
        # [外] が引き起こすエラーを回避する
        invalid_rids = df[df['hname'].str.contains('[外]')].loc[:, 'rid'].unique()
        df = df[~df['rid'].isin(invalid_rids)]
        # Post position の正規化
        df = df[df['rid'].isin(df['rid'].unique())]
        n_post = StandardScaler().fit_transform(df.loc[:, 'post'].values.reshape(-1, 1))
        df['n_post'] = np.ravel(n_post)
        # Extraction: X, w
        X = []
        w = []
        race_ids = []
        variables = ['n_presi', 'n_avgsi4', 'n_disavgsi', 'n_goavgsi', 'w2c', 'eps',
                'jnowin', 'jwinper', 'jst1miss', 'newdis', 'n_post']
        for rid, rdf in df.groupby('rid'):
            try:
                w_i = rdf.loc[:, 'fp'].values.tolist().index(1)
                w.append(w_i)
                race_ids.append(rid)
                X_i = np.asmatrix(rdf[variables].values)
                X.append(X_i)
            except ValueError:
                print(f'http://jiro8.sakura.ne.jp/index.php?code={rid}')
        # Assertion
        self.assertEqual(type(X), list)
        self.assertEqual(type(X[0]), np.matrix)
        self.assertEqual(X[0].shape[1], len(variables))
        self.assertEqual(len(X), len(w))
        self.assertEqual(len(w), len(race_ids))
        # Estimation
        X_train, w_train = X[:10000], w[:10000]
        X_test, w_test = X[10000:], w[10000:]
        cl = ConditionalLogit(use_bias=True, eta=1e-5, max_iter=30)
        cl.estimate(X_train, np.array(w_train), batch_size=20)
        print(cl.log_likelihood)
        for col, coef_ in zip(['bias'] + variables, cl.beta):
            print(f'{col}: {round(coef_, 2)}')

        # Prediction
        win_probas = cl.predict(X_test)
        for rid, win_proba in zip(race_ids[10000:], win_probas):
            for i, (hname, row) in enumerate(df[df['rid'] == rid].groupby('hname')):
                print(f"{rid} | {row['draw']} {hname} {row['odds']} {row['fp']} : {win_proba[i]}")

if __name__ == '__main__':
    unittest.main()
