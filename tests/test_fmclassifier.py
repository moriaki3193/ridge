# -*- coding: utf-8 -*-
import unittest
import numpy as np
import pandas as pd
from os import path
from ridge.models import FMClassifier
from sklearn.metrics import accuracy_score


BASEDIR = path.dirname(path.abspath(__file__))


class TestFMClassifier(unittest.TestCase):
    """Testing FMClassifier with the Titanic dataset.
    """

    def setUp(self):

        def impute_age(age_mean):
            def _impute_age(x):
                if x.Sex == 'male':
                    return round(age_mean['male'])
                elif x.Sex == 'female':
                    return round(age_mean['female'])
            return _impute_age

        train_df = pd.read_csv(path.join(BASEDIR, 'data', 'titanic-train.csv'))
        test_df = pd.read_csv(path.join(BASEDIR, 'data', 'titanic-test.csv'))
        train_df = train_df.drop(['Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
        test_df = test_df.drop(['Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)

        # [START Age Imputation]
        train_age_mean = train_df.groupby('Sex').Age.mean()
        test_age_mean = test_df.groupby('Sex').Age.mean()
        train_df.Age.fillna(train_df[train_df.Age.isnull()].apply(impute_age(train_age_mean), axis=1), inplace=True)
        test_df.Age.fillna(test_df[test_df.Age.isnull()].apply(impute_age(test_age_mean), axis=1), inplace=True)
        # [END Age Imputation]

        # [START One-hot vectorization]
        train_df['Female'] = train_df['Sex'].map({'male': 0, 'female': 1}).astype(int)
        test_df['Female'] = test_df['Sex'].map({'male': 0, 'female': 1}).astype(int)

        pclass_train_df = pd.get_dummies(train_df['Pclass'], prefix=('Class'))
        pclass_test_df = pd.get_dummies(test_df['Pclass'], prefix=('Class'))
        pclass_train_df = pclass_train_df.drop(['Class_3'], axis=1)
        pclass_test_df = pclass_test_df.drop(['Class_3'], axis=1)
        train_df = train_df.join(pclass_train_df)
        test_df = test_df.join(pclass_test_df)
        # [END One-hot vectorization]

        self.train = train_df
        self.test = test_df

    def test_fitting_fmclassifier(self):
        X_train = self.train.drop(['PassengerId', 'Survived', 'Pclass', 'Sex'], axis=1).values
        y_train = self.train.Survived.values
        X_test = self.test.drop(['PassengerId', 'Pclass', 'Sex'], axis=1).values
        pids = self.test.PassengerId.values
        model = FMClassifier().fit(X_train, y_train, k=3, l2=1e-1, eta=1e-2, n_iter=300)
        y_pred = model.predict(X_test, target='0-1')
        # Make Kaggle Submission Data
        pd.DataFrame({'PassengerId': pids, 'Survived': y_pred}).to_csv(path.join(BASEDIR, 'tmp', 'titanic-result.csv'), index=None)
        # Print Loss Series
        print(model.loss_series)


if __name__ == '__main__':
    unittest.main()