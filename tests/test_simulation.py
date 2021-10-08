import numpy as np

from pytest import mark
from protosc.simulation import create_independent_data, create_correlated_data,\
    create_categorical_data, compare_results
from protosc.feature_matrix import FeatureMatrix
from protosc.model.utils import train_kfold_validate


@mark.parametrize("n_features,n_samples,n_true_features", [
    (100, 500, 20),
    (55, 421, 15),
    (21, 345, 3),
])
def test_independent_data(n_features, n_samples, n_true_features):
    X, y, truth = create_independent_data(n_features, n_samples,
                                          n_true_features)
    check_data(X, y, truth, n_features, n_samples)
    assert len(truth["selected_features"]) == n_true_features


@mark.parametrize(
    "n_base_features,n_samples,n_true_features,n_feature_correlated", [
        (50, 450, 10, 5),
        (23, 728, 5, 2),
        (13, 354, 6, 1),
    ])
def test_correlated_data(n_base_features, n_samples, n_true_features,
                         n_feature_correlated):
    X, y, truth = create_correlated_data(
        n_base_features, n_samples, n_true_features, n_feature_correlated)
    check_data(X, y, truth, n_base_features*n_feature_correlated, n_samples)
    assert len(truth["selected_features"]) == n_true_features*n_feature_correlated
    assert len(np.unique(truth["clusters"])) == n_base_features
    assert len(truth["clusters"]) == n_base_features*n_feature_correlated

    if n_feature_correlated > 1:
        clust_features = np.where(truth["clusters"] == 0)[0]
        cov = np.corrcoef(X[:, clust_features], rowvar=False)
        assert np.all(cov > 0.5)


@mark.parametrize(
    "n_features,n_samples,n_true_features,n_categories", [
        (50, 239, 10, 5),
        (23, 581, 5, 2),
        (12, 349, 8, 3),
    ]
)
def test_categorical_data(n_features, n_samples, n_true_features,
                          n_categories):
    X, y, truth = create_categorical_data(
        n_features, n_samples, n_true_features, n_categories,
        min_dev=0.5, max_dev=1.0)
    check_data(X, y, truth, n_features, n_samples)
    assert len(np.unique(y)) == n_categories
    assert len(truth["selected_features"]) == n_true_features


def check_data(X, y, truth, n_features, n_samples):
    assert isinstance(X, FeatureMatrix)
    assert X.shape == (n_samples, n_features)
    assert len(y) == n_samples

    accuracy = train_kfold_validate(X, y, truth["selected_features"])
    random_features = np.delete(np.arange(n_features), truth["selected_features"])
    random_selection = np.random.choice(
        random_features, 
        size=min(len(random_features), len(truth["selected_features"])),
        replace=False)
    random_accuracy = train_kfold_validate(X, y, random_selection)
    assert accuracy > random_accuracy
    if "biases" in truth:
        assert set(np.where(truth["biases"])[0]) == set(truth["selected_features"])


def test_compare():
    _, _, truth = create_independent_data()
    output = compare_results(truth["selected_features"], truth)
    assert np.allclose(np.array(list(output.values())), 1)
    output = compare_results([], truth)
    assert np.allclose(np.array(list(output.values())), 0)
