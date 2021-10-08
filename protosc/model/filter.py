from protosc.model.utils import select_features
from protosc.model.utils import train_xvalidate
from protosc.model.base import BaseFoldModel


class FilterModel(BaseFoldModel):
    def __init__(self, n_fold=8):
        self.n_fold = n_fold

    def copy(self):
        return self.__class__(self.n_fold)

    def _execute_fold(self, fold):
        X_train, y_train, X_val, y_val = fold
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
