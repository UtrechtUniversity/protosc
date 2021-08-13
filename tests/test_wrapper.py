""" Tests wrapper method """

from protosc.wrapper import Wrapper
from protosc.simulation import create_correlated_data
from protosc.filter_model import select_features
import numpy as np

N = 2
ADD_IM = True
FOLD_SEED = None


def __create_data():
    np.random.seed(1928374)
    X, y, _ = create_correlated_data()
    _, clusters = select_features(X, y, chisq_threshold=0.25)
    return X, y, clusters


def __run_wrapper():
    X, y, clusters = __create_data()
    fast = Wrapper(X, y, clusters, add_im=ADD_IM, fold_seed=FOLD_SEED)
    output = fast.wrapper(n_rounds=N, n_jobs=-1)
    return output


def __test_model(output):
    model = output['model']
    clusters = output['clusters']
    assert isinstance(model, list)
    assert len(model) == N
    assert all([isinstance(i, list) for i in model])
    assert all([len(model[i]) == len(clusters[i]) for i in range(N)])


def __test_features(output):
    features = output['features']
    assert isinstance(features, list)
    assert len(features) == N
    assert all([isinstance(i, np.ndarray) for i in features])
    assert all(
        [len(np.unique(features[i])) == len(features[i]) for i in range(N)])


def __test_clusters(output):
    clusters = output['clusters']
    assert isinstance(clusters, list)
    assert len(clusters) == N
    assert all([isinstance(i, list) for i in clusters])


def __test_accuracy(output):
    accuracy = output['accuracy']
    assert isinstance(accuracy, list)
    assert len(accuracy) == N
    assert all([isinstance(i, float) for i in accuracy])


def __test_recurring(output):
    recurring = output['recurring']
    new = []
    [new.extend(i) for i in output['clusters']]
    assert isinstance(recurring, list)
    assert [j for j in recurring if new.count(j) == N] == recurring


def test_wrapper():
    output = __run_wrapper()
    __test_model(output)
    __test_features(output)
    __test_clusters(output)
    __test_accuracy(output)
    if N > 1:
        __test_recurring(output)
