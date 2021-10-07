from collections import defaultdict

import numpy as np

from protosc.feature_matrix import FeatureMatrix
from protosc.model.final_selection import final_selection
from protosc.parallel import execute_parallel
from protosc.model.utils import select_features
from protosc.model.utils import train_xvalidate
from protosc.model.base import BaseModel


class FilterModel(BaseModel):
    def __init__(self, n_fold=8, null_distribution=False):
        self.n_fold = n_fold
        self.null_distribution = null_distribution

    def copy(self):
        return self.__class__(self.n_fold, self.null_distribution)

    def _execute_fold(self, fold):
        X_train, y_train, X_val, y_val = fold

        # Select the top n features needed to make .25
        if self.null_distribution:
            np.random.shuffle(y_train)

        selected_features, _ = select_features(X_train, y_train)

        # Build the SVM model with specified kernel ('linear', 'rbf',
        # 'poly', 'sigmoid') using only selected features
        accuracy = train_xvalidate(
            X_train[:, selected_features], y_train,
            X_val[:, selected_features], y_val)
        return {"features": selected_features, "accuracy": accuracy}


def compute_filter_fold(cur_fold):
    X_train, y_train, _X_val, _y_val = cur_fold

    # Select the top n features needed to make .25
    selected_features, clusters = select_features(X_train, y_train)
    return {
        "selected_features": selected_features,
        "clusters": clusters,
        "cur_fold": cur_fold,
    }


def compute_filter_data(X, y, n_fold, fold_seed=None, seed=None):
    if not isinstance(X, FeatureMatrix):
        X = FeatureMatrix(X)

    fold_rng = np.random.default_rng(fold_seed)

    # Split data into 8 partitions: later use 1 partition as
    # validating data, other 7 as train data

    # Train an SVM on the train set while using the selected features
    # (i.e., making up 25% of chisquare scores), crossvalidate on holdout
    filter_data = []

    np.random.seed(seed)
    for cur_fold in X.kfold(y, k=n_fold, rng=fold_rng):
        X_train, y_train, _X_val, _y_val = cur_fold

        # Select the top n features needed to make .25
        selected_features, clusters = select_features(X_train, y_train)
        filter_data.append({
            "selected_features": selected_features,
            "clusters": clusters,
            "cur_fold": cur_fold,
        })

    return filter_data
