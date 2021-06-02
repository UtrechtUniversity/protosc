from protosc.simulation import create_correlated_data
from protosc.filter_model import filter_model
from collections import defaultdict
from protosc.final_selection import final_selection


def test_final_selection():
    X, y, truth = create_correlated_data(
        n_base_features=30, n_true_features=4, n_examples=300,
        n_feature_correlated=3)
    feature_accuracy = filter_model(X, y, fold_seed=213874)
    null_accuracy = defaultdict(lambda: [])
    for _ in range(100):
        res = filter_model(X, y, fold_seed=213874, null_distribution=True)
        for i, val in enumerate(res):
            null_accuracy[i].append(val[1])
    null_accuracy = list(null_accuracy.values())

    feature_selection = final_selection(feature_accuracy, null_accuracy)
    assert len(set(feature_selection)-set(truth["selected_features"])) == 0
    assert len(feature_selection) % 3 == 0
    assert len(feature_selection) > 0
