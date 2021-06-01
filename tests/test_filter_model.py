import numpy as np
from protosc.filter_model import select_fold, train_xvalidate, calc_chisquare,\
    fast_chisquare, create_clusters, select_features, filter_model
from protosc.simulation import create_correlated_data, create_simulation_data


def get_test_matrix(n_row=100, n_col=50):
    X = np.zeros((n_row, n_col))
    X = X + np.arange(n_row).reshape(n_row, 1)
    X = X + np.arange(n_col).reshape(1, n_col)/1000
    y = np.random.randint(2, size=n_row)
    return X, y


def test_select_fold():
    n_fold = 5
    n_row = 100
    n_col = 50
    X, y = get_test_matrix(n_row, n_col)
    X_folds = np.array_split(X, n_fold)
    y_folds = np.array_split(y, n_fold)
    rng = np.random.default_rng()
    for i_fold in range(n_fold):
        X_train, y_train, X_val, y_val = select_fold(
            X_folds, y_folds, i_fold, rng, balance=False)
        assert np.allclose(X_train.shape, ((n_fold-1)/n_fold*n_row, n_col))
        assert len(y_train) == X_train.shape[0]
        assert np.allclose(X_val.shape, 1/n_fold*n_row, n_col)
        assert len(y_val) == X_val.shape[0]
        assert len(np.unique(X_train)) == X_train.size
        assert len(np.unique(X_val)) == X_val.size

        X_train, y_train, X_val, y_val = select_fold(
            X_folds, y_folds, i_fold, rng, balance=True)
        assert np.sum(y_train) == len(y_train)/2
        assert np.sum(y_val) == len(y_val)/2
        assert len(np.unique(X_train)) == X_train.size
        assert len(np.unique(X_val)) == X_val.size

        assert isinstance(
            train_xvalidate(X_train, y_train, X_val, y_val), dict)


def test_kruskal():
    X, y = get_test_matrix()
    assert np.allclose(calc_chisquare(X, y), fast_chisquare(X, y))


def test_select_clusters():
    X, _, truth = create_correlated_data()

    features_sorted = np.random.permutation(X.shape[1])
    cluster_groups = create_clusters(features_sorted, X)
    for cluster in cluster_groups:
        assert np.all(np.array(
            truth["clusters"][cluster]) == truth["clusters"][cluster][0])


def test_select_features():
    X, y, _ = create_simulation_data()
    assert isinstance(select_features(X, y, fast_chisq=False), list)
    assert isinstance(select_features(X, y, fast_chisq=True), list)


def test_filter_model():
    X, y, _ = create_simulation_data()
    X[:10, 0] = 1
    output = filter_model(X, y)
    assert isinstance(output, list)
