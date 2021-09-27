from abc import ABC, abstractmethod

import numpy as np

from protosc.feature_matrix import FeatureMatrix
from protosc.parallel import execute_parallel


class BaseModel(ABC):
    def __init__(self, n_fold=8):
        self.n_fold = n_fold

    def execute(self, X, y, fold_seed=None, seed=None, n_jobs=1,
                progress_bar=False):
        if not isinstance(X, FeatureMatrix):
            X = FeatureMatrix(X)
        fold_rng = np.random.default_rng(fold_seed)
        np.random.seed(seed)
        seeds = [np.random.randint(0, 91827384) for _ in range(self.n_fold)]

        jobs = []
        for fold in X.kfold(y, k=self.n_fold, rng=fold_rng):
            jobs.append({
                "fold": fold,
                "seed": seeds[len(jobs)],
            })
        return execute_parallel(jobs, parallel_model, n_jobs=n_jobs,
                                progress_bar=progress_bar,
                                args=[self.copy()])

    @abstractmethod
    def _execute_fold(self, fold):
        raise NotImplementedError

    def copy(self):
        return self.__class__(n_fold=self.n_fold)


def parallel_model(model, fold, seed):
    np.random.seed(seed)
    return model._execute_fold(fold)
