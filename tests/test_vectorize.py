# -*- coding: utf-8 -*-
import pickle
import unittest
import numpy as np
import pandas as pd
from os import path
from tqdm import tqdm
from scipy import sparse


GROUP_KEY = 'rid'
TARGET_KEY = 'fp_std'
BASEDIR = path.dirname(path.abspath(__file__))
PATH2FEATURES = path.join(BASEDIR, 'data', 'features.csv')
FEATURES_OUTPUT = path.join(BASEDIR, 'tmp', 'horseracing_sparse_features.npz')
TARGETS_OUTPUT = path.join(BASEDIR, 'tmp', 'horseracing_targets.npy')
RACEIDS_OUTPUT = path.join(BASEDIR, 'tmp', 'horseracing_raceids.pkl')
TRAIN_SIZE = 0.8


class TestVectorize(unittest.TestCase):

    def test_construct_sparse_matrix(self):
        """Construct a sparse matrices of the horse racing dataset.

        What will be saved after running this suite
        -------------------------------------------
        """
        df = pd.read_csv(PATH2FEATURES)
        df.astype({GROUP_KEY: str})
        unique_hnames = df['hname'].unique()
        hname2ind = pd.get_dummies(unique_hnames)
        invalid_rids = df[df['odds'] == 0][GROUP_KEY].unique()
        context_cols = ['n_presi', 'n_avgsi4', 'n_disavgsi', 'n_goavgsi',
                        'w2c', 'eps', 'draw', 'newdis',
                        'jnowin', 'jwinper', 'jst1miss']

        # [START Construct Competitor & Entity Index Vector]
        n_horses = len(unique_hnames)
        indexing_features = sparse.coo_matrix((0, 2 * n_horses), dtype=np.int8)
        grouped = df[~df[GROUP_KEY].isin(invalid_rids)].groupby(GROUP_KEY)

        pbar = tqdm(total=len(grouped))
        pbar.set_description('Constructing Indexing Matrix...')
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

            indexing_features = sparse.vstack([indexing_features, index_matrix])  # TODO the greater idx is, the slower...
            pbar.update(1)
        pbar.close()
        # [END Construct Competitor & Entity Index Vector]

        # [START Construct Context Vector & Target]
        n_train_rows = 0  # ← MAX Train data index
        n_contexts = len(context_cols)
        context_features = sparse.coo_matrix((0, n_contexts))
        target_series = []
        raceid_series = []
        pbar2 = tqdm(total=len(grouped))
        pbar2.set_description('Constructing Context Matrix ...')
        for idx, (_, rdata) in enumerate(grouped):
            context_matrix = sparse.coo_matrix(rdata[context_cols].values)
            context_features = sparse.vstack([context_features, context_matrix])
            target_series += rdata[TARGET_KEY].values.tolist()
            raceid_series += rdata[GROUP_KEY].values.tolist()
            # [START Get Train Test Split Index]
            if idx == np.round(TRAIN_SIZE * len(grouped)):
                n_train_rows, _ = context_features.shape
            # [END Get Train Test Split Index]
            pbar2.update(1)
        pbar2.close()
        # [END Construct Context Vector & Target]


        # Finally, concat indexing_features & context_features
        features = sparse.hstack((indexing_features, context_features))
        target_series = np.asarray(target_series)

        # Display Stats
        print('---' * 20)
        print('Summary')
        print(f'+ The number of races: {len(grouped)}')
        print(f'+ The number of horses: {n_horses}')
        print('Indexing Matrix Stats')
        print(f'+ Shape of sparse matrix: {indexing_features.shape}')
        print(f'+ The number of nonzero elems: {indexing_features.nnz}')
        print('Context Matrix Stats')
        print(f'Shape of dense matrix: {context_features.shape}')
        print('Train Test Split')
        print(f'+ Maximum Train data row index: {n_train_rows}')
        print('---' * 20)

        # Unit Testing
        self.assertEqual(indexing_features.dtype, np.int8)
        self.assertEqual(indexing_features.shape[0], context_features.shape[0])
        self.assertEqual(features.shape[1], indexing_features.shape[1] + context_features.shape[1])
        self.assertEqual(features.shape[0], len(target_series))
        self.assertEqual(len(target_series), len(raceid_series))

        # Save the features & targets
        sparse.save_npz(FEATURES_OUTPUT, features)
        np.save(TARGETS_OUTPUT, target_series)
        with open(RACEIDS_OUTPUT, mode='wb') as fp:
            pickle.dump(raceid_series, fp)


if __name__ == '__main__':
    unittest.main()
