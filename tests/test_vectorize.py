# -*- coding: utf-8 -*-
import unittest
import numpy as np
import pandas as pd
from os import path
from tqdm import tqdm
from scipy import sparse


GROUP_KEY = 'rid'
PATH2FEATURES = path.join(path.dirname(path.abspath(__file__)), 'data', 'features.csv')
PATH2OUTPUT = path.join(path.dirname(path.abspath(__file__)), 'tmp', 'sparse_features')


class TestVectorize(unittest.TestCase):

    def test_construct_sparse_matrix(self):
        df = pd.read_csv(PATH2FEATURES)
        unique_hnames = df['hname'].unique()
        hname2ind = pd.get_dummies(unique_hnames)
        invalid_rids = df[df['odds'] == 0][GROUP_KEY].unique()

        # [START Construct Competitor & Entity Index Vector]
        n_horses = len(unique_hnames)
        features = sparse.coo_matrix((n_horses, 0), dtype=np.int8)
        grouped = df[~df[GROUP_KEY].isin(invalid_rids)].groupby(GROUP_KEY)
        pbar = tqdm(total=len(grouped))
        for idx, (_, rdata) in enumerate(grouped):
            entries = rdata['hname']
            n_entries = len(entries)
            val = np.ones(np.power(n_entries, 2))  # shape = (n_entries ** 2, )
            row = np.array([np.repeat(i, n_entries) for i in np.arange(0, n_entries)]).flatten()  # shape = (n_entries ** 2, )
            entry_indices = np.array([hname2ind[hname2ind[hname] == 1].index[0] for hname in entries])

            # [START Obtain Column Indices]
            col = []
            for jdx, entry_index in enumerate(entry_indices):
                _copy = entry_indices.copy()
                combination_indices = np.delete(entry_indices, jdx)
                entity_index = entry_index + n_horses  # entity & entry
                this_col = np.append(combination_indices, entity_index)
                col += this_col.tolist()
            col = np.array(col)
            # [END Obtain Column Indices]

            index_matrix = sparse.coo_matrix((val, (row, col)), shape=(n_entries, 2 * n_horses), dtype=np.int8)

            # [START Assertion]
            if idx == len(grouped):
                for hname in entries:
                    nonzeros = index_matrix.toarray().nonzero()
                    target_ind = hname2ind[hname2ind[hname] == 1].index[0]
                    _index = nonzeros[0][n_entries - 1]
                    ind_as_entity = index_matrix.toarray().nonzero()[1][_index]
                    self.assertEqual(target_ind, ind_as_entity - n_horses)
            # [END Assertion]

            features = sparse.vstack([features, index_matrix])  # TODO the greater idx is, the slower...
            pbar.update(1)
        pbar.close()
        # [END Construct Competitor & Entity Index Vector]

        self.assertEqual(features.dtype, np.int8)
        np.save(PATH2OUTPUT, features)


if __name__ == '__main__':
    unittest.main()
