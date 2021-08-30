""" Tests wrapper method """

from protosc.wrapper import Wrapper
from protosc.simulation import create_correlated_data
import numpy as np

N_FOLD = 8
ADD_IM = True
FOLD_SEED = None


def __create_data():
    np.random.seed(1928374)
    X, y, _ = create_correlated_data()
    return X, y


def __run_wrapper():
    X, y = __create_data()
    fast = Wrapper(X, y, n_fold=N_FOLD, add_im=ADD_IM, fold_seed=FOLD_SEED)
    output = fast.wrapper(n_jobs=-1)
    return output


def __test_model(output):
    model = output['model']
    assert isinstance(model, list)
    assert len(model) == N_FOLD
    assert all([isinstance(i, list) for i in model])


def __test_features(output):
    features = output['features']
    assert isinstance(features, list)
    assert len(features) == N_FOLD
    assert all([isinstance(i, np.ndarray) for i in features])
    assert all([len(
            np.unique(features[i])) == len(features[i]) for i in range(N_FOLD)
            ])


def __test_accuracy(output):
    accuracy = output['accuracy']
    assert isinstance(accuracy, list)
    assert len(accuracy) == N_FOLD
    assert all([isinstance(i, float) for i in accuracy])


def __test_recurring(output):
    recurring = output['recurring']
    new = []
    [new.extend(i) for i in output['features']]
    assert isinstance(recurring, list)
    assert [j for j in recurring if new.count(j) == N_FOLD] == recurring


def test_wrapper():
    output = __run_wrapper()
    __test_model(output)
    __test_features(output)
    __test_accuracy(output)
    if N_FOLD > 1:
        __test_recurring(output)
