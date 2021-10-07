""" Tests wrapper method """
import numpy as np
import pytest

from protosc.model.wrapper import Wrapper
from protosc.simulation import create_correlated_data


@pytest.fixture
def easy_data():
    np.random.seed(1928374)
    return create_correlated_data(n_base_features=10, n_true_features=5,
                                  n_examples=100, min_dev=20, max_dev=30,
                                  n_feature_correlated=2)


@pytest.fixture
def X(easy_data):
    return easy_data[0]


@pytest.fixture
def y(easy_data):
    return easy_data[1]


@pytest.fixture
def truth(easy_data):
    return easy_data[2]


@pytest.mark.parametrize('n_fold', [3, 8])
@pytest.mark.parametrize('max_features', [10, 30])
@pytest.mark.parametrize('search_fraction', [0.8, 1.0])
@pytest.mark.parametrize('reversed_clusters', [True, False])
@pytest.mark.parametrize('greedy', [True, False])
@pytest.mark.parametrize('exclusion_step', [True, False])
@pytest.mark.parametrize('max_nop_rounds', [4, 6])
def test_wrapper(X, y, truth, n_fold, max_features, search_fraction,
                 reversed_clusters, greedy, exclusion_step, max_nop_rounds):
    assert isinstance(truth, dict)
    wrapper = Wrapper(n_fold, max_features, search_fraction, reversed_clusters,
                      greedy, exclusion_step, max_nop_rounds)
    model_output = wrapper.execute(X, y, fold_seed=1298374)
    check_model_output(model_output, n_fold=n_fold)


def check_model_output(output, n_fold):
    assert isinstance(output, list)
    assert len(output) == n_fold
    for fold_output in output:
        assert isinstance(fold_output, dict)
        assert "features" in fold_output
        features = fold_output["features"]
        assert isinstance(features, (list, np.ndarray))
        if len(features):
            assert isinstance(features[0], (int, np.int, np.int64))
        assert len(features) == len(set(features))

        assert "accuracy" in fold_output
        assert isinstance(fold_output["accuracy"], float)
