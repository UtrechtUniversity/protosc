""" Tests wrapper method """

from protosc.model.wrapper import Wrapper
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
    return Wrapper(n_fold=N_FOLD, add_im=ADD_IM).execute(
        X, y, fold_seed=FOLD_SEED)


def __test_model(output):
    assert isinstance(output, list)
    assert len(output) == N_FOLD
    assert all([isinstance(i, dict) for i in output])


def __test_features(output):
    features = [x["features"] for x in output]
    assert isinstance(features, list)
    assert len(features) == N_FOLD
    assert all([isinstance(i, (np.ndarray, list)) for i in features])
    assert all([len(
            np.unique(features[i])) == len(features[i]) for i in range(N_FOLD)
            ])


def __test_accuracy(output):
    accuracy = [x["accuracy"] for x in output]
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
