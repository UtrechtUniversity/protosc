from abc import ABC, abstractmethod

import numpy as np

from protosc.feature_matrix import FeatureMatrix


class BaseModel(ABC):
    def __init__(self, n_fold=8):
        self.n_fold = n_fold

    def execute(self, X, y, fold_seed=None, seed=None):
        if not isinstance(X, FeatureMatrix):
            X = FeatureMatrix(X)
        fold_rng = np.random.default_rng(fold_seed)
        np.random.seed(seed)
        results = []
        for fold in X.kfold(y, k=self.n_fold, rng=fold_rng):
            results.append(self._execute_fold(fold))
        return results

    @abstractmethod
    def _execute_fold(self, fold):
        raise NotImplementedError
