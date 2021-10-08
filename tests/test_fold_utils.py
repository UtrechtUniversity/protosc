import numpy as np

from protosc.model.utils import train_xvalidate, create_clusters, select_features
from protosc.model.filter import FilterModel
from protosc.simulation import create_correlated_data, create_independent_data
from protosc.feature_matrix import FeatureMatrix


def get_test_matrix(n_row=100, n_col=50):
    X = np.zeros((n_row, n_col))
    X = X + np.arange(n_row).reshape(n_row, 1)
    X = X + np.arange(n_col).reshape(1, n_col)/1000
    y = np.random.randint(2, size=n_row)
    return FeatureMatrix(X), y


def test_select_fold():
    n_fold = 5
    n_row = 100
    n_col = 50
    X, y = get_test_matrix(n_row, n_col)
    rng = np.random.default_rng()
    for X_train, y_train, X_val, y_val in X.kfold(y, n_fold, rng, balance=False):
        assert np.allclose(X_train.shape, ((n_fold-1)/n_fold*n_row, n_col))
        assert len(y_train) == X_train.shape[0]
        assert np.allclose(X_val.shape, 1/n_fold*n_row, n_col)
        assert len(y_val) == X_val.shape[0]
        assert len(np.unique(X_train[:])) == X_train.size
        assert len(np.unique(X_val[:])) == X_val.size

    for X_train, y_train, X_val, y_val in X.kfold(y, n_fold, rng, balance=True):
        assert np.sum(y_train) == len(y_train)/2
        assert np.sum(y_val) == len(y_val)/2
        assert len(np.unique(X_train[:])) == X_train.size
        assert len(np.unique(X_val[:])) == X_val.size

        assert isinstance(
            train_xvalidate(X_train[:], y_train, X_val[:], y_val), float)


def test_select_clusters():
    X, _, truth = create_correlated_data()

    X = FeatureMatrix.from_matrix(X)
    features_sorted = np.random.permutation(X.shape[1])
    cluster_groups = create_clusters(features_sorted, X)
    for cluster in cluster_groups:
        assert np.all(np.array(
            truth["clusters"][cluster]) == truth["clusters"][cluster][0])


def test_select_features():
    X, y, _ = create_independent_data()
    selected_features, clusters = select_features(X, y)
    assert isinstance(selected_features, list)
    assert isinstance(clusters, list)


