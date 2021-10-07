import numpy as np
import pytest

from protosc.model.wrapper import WrapperModel, ClusteredSelection
from protosc.model.filter import FilterModel
from protosc.simulation import create_correlated_data
from protosc.model.random import RandomModel
from protosc.model.pseudo_random import PseudoRandomModel
from protosc.model.combined_fold import CombinedFoldModel
from _collections import defaultdict
from protosc.model.utils import compute_accuracy


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
    np.random.seed(298374)
    wrapper = WrapperModel(
        n_fold, max_features, search_fraction, reversed_clusters,
        greedy, exclusion_step, max_nop_rounds)
    selection = wrapper.execute(X, y, fold_seed=1298374)
    check_interim_output(wrapper.interim_data, n_fold=n_fold)
    assert set(selection) == set(truth["selected_features"])
    assert len(set(selection)) == len(selection)


def test_wrapper_remove():
    np.random.seed(192837)
    X, y, truth = create_correlated_data(n_base_features=50, n_true_features=5,
                                         n_examples=500, min_dev=0.2, max_dev=0.4,
                                         n_feature_correlated=2)
    fold_rng = np.random.default_rng(102128434)
    fold = [x for x in X.kfold(y, k=8, rng=fold_rng)][0]
    wrapper = WrapperModel(8, exclusion_step=True)
    all_clusters = [[x] for x in range(X.shape[1])]
    clusters = np.random.choice(np.arange(len(all_clusters)), size=len(all_clusters)//2, replace=False)
    selection = ClusteredSelection(all_clusters, clusters)
    accuracy = compute_accuracy(fold, selection.features)
    new_selection, new_accuracy = wrapper._remove_procedure(fold, selection, accuracy)
    assert new_accuracy > accuracy
    new_n_false = len(set(new_selection.features) - set(truth["selected_features"]))
    old_n_false = len(set(selection.features) - set(truth["selected_features"]))
    assert new_n_false < old_n_false


@pytest.mark.parametrize('n_fold', [2, 6, 8])
def test_filter(X, y, truth, n_fold):
    filter_model = FilterModel(n_fold=n_fold)
    check_fold_model(filter_model, X, y, truth, n_fold)


@pytest.mark.parametrize('n_fold', [2, 6, 8])
def test_random(X, y, truth, n_fold):
    model = RandomModel(n_fold)
    check_fold_model(model, X, y, truth, n_fold, check_perf=False)


@pytest.mark.parametrize('n_fold', [2, 6, 8])
def test_pseudo_random(X, y, truth, n_fold):
    model = PseudoRandomModel(n_fold)
    check_fold_model(model, X, y, truth, n_fold, check_perf=False)


@pytest.mark.parametrize('n_fold', [2, 6, 8])
def test_combined(X, y, truth, n_fold):
    model = CombinedFoldModel(n_fold)
    all_selection = model.execute(X, y, fold_seed=1298374)
    reformed_data = defaultdict(lambda: [])
    for package in model.interim_data:
        for name, res in package.items():
            if name != "null_distribution":
                reformed_data[name].append(res)
    assert len(reformed_data) == 5
    for name, res in reformed_data.items():
        selection = all_selection[name]
        assert len(set(selection)) == len(selection)
        if name not in ["random", "pseudo_random"]:
            assert set(selection) == set(truth["selected_features"])
        check_interim_output(res, n_fold)


def check_fold_model(model, X, y, truth, n_fold, check_perf=True):
    selection = model.execute(X, y, fold_seed=1298374)
    check_interim_output(model.interim_data, n_fold=n_fold)
    assert len(set(selection)) == len(selection)
    if check_perf:
        assert set(selection) == set(truth["selected_features"])


def check_interim_output(output, n_fold):
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
