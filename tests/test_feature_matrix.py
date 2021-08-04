from protosc.pipeline import BasePipeElement
import numpy as np
from protosc.pipe_complex import PipeComplex
from protosc.feature_matrix import FeatureMatrix
from libpasteurize.fixes.feature_base import Feature
import pytest


class RandomPipe(BasePipeElement):
    def __init__(self, dim):
        self.dim = dim

    def _execute(self, _):
        try:
            return np.random.randn(self.dim)
        except TypeError:
            return np.random.randn(*self.dim)

    @property
    def name(self):
        return f"random_{str(self.dim)}"


@pytest.mark.parametrize(
    "n_samples", [
        200,
        20
    ]
)
def test_feature_matrix(n_samples):
    pc = PipeComplex(RandomPipe((20, 5)), RandomPipe(10))
    results = pc.execute([None]*n_samples)
    fm = FeatureMatrix.from_pipe_data(results)
    assert fm.shape == (n_samples, 30)
    assert fm.X.shape == (n_samples, 110)
    fm_copy = fm.copy()
    fm_copy.X[0, 0] = 0
    assert fm_copy.X[0, 0] != fm.X[0, 0]
    assert fm_copy.shape == fm.shape
    assert fm_copy.X.shape == fm.X.shape

    assert fm.corrcoef(np.random.choice(20, size=10, replace=False)).shape == (10, 10)

    assert fm[:10, :10].shape == (10, 50)
    assert fm[-5:, -5:].shape == (5, 5)
    assert fm[8:12, 18:22].shape == (4, 12)

    assert np.allclose(fm[:, -1], fm.X[:, -1].reshape(-1, 1))

    fm.add_random_columns(10)
    assert fm.shape == (n_samples, 40)
    assert fm.X.shape == (n_samples, 120)

    assert fm.size == n_samples*40
    y = (2*np.random.rand(n_samples)).astype(int)
    i_fold = 0
    for X_train, y_train, X_val, y_val in fm.kfold(y, k=8, balance=True):
        assert len(y_train) == 2*np.sum(y_train) or len(y_train) == np.sum(y_train) or np.sum(y_train) == 0
        assert len(y_val) == 2*np.sum(y_val) or len(y_val) == np.sum(y_val) or np.sum(y_val) == 0
        assert isinstance(X_train, FeatureMatrix)
        assert isinstance(X_val, FeatureMatrix)
        assert X_train.shape[1] == fm.shape[1]
        assert X_val.shape[1] == fm.shape[1]
        assert X_val.shape[0] < fm.shape[0]
        assert X_train.shape[0] < fm.shape[0]
        i_fold += 1
    assert i_fold == 8
