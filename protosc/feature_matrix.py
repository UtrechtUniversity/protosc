import numpy as np


class FeatureMatrix():
    def __init__(self, X, rev_lookup_table):
        self.X = X
        self.rev_lookup_table = rev_lookup_table

    @classmethod
    def from_pipe_data(cls, data):
        n_row = len(data)
        # First find the structure
        rev_lookup_table = []
        n_col = 0
        for key, features in data[0].items():
            for sub_feature_id in range(features.shape[0]):
                rev_lookup_table.append({
                    "pipeline": key,
                    "sub_feature_id": sub_feature_id,
                    "col_ids": [i+n_col for i in range(features.shape[1])]
                })
                n_col += features.shape[1]

        X = np.zeros((n_row, n_col))
        for i_data, data_packet in enumerate(data):
            i_col = 0
            for features in data_packet.values():
                for sub_feature_id in range(features.shape[0]):
                    X[i_data, i_col: i_col+features.shape[1]] = features[
                        sub_feature_id]
                    i_col += features.shape[1]
        return cls(X, rev_lookup_table)

    @classmethod
    def from_matrix(cls, X):
        rev_lookup_table = [
            {"pipeline": "matrix",
             "sub_feature_id": i,
             "col_ids": [i]} for i in range(X.shape[1])
        ]
        return cls(X, rev_lookup_table)

    def __getitem__(self, key):
        return self.get_slice(key)

    def corrcoef(self, features_sorted):
        X_new = self[:, features_sorted]
        r_matrix = np.corrcoef(X_new, rowvar=False)
        cols_occ = [len(self.rev_lookup_table[x]["col_ids"])
                    for x in features_sorted]

        final_r_matrix = np.zeros((len(features_sorted), len(features_sorted)))
        cols_cum = np.append([0], np.cumsum(cols_occ))
        for i_feature in range(len(features_sorted)):
            i_start, i_end = cols_cum[i_feature], cols_cum[i_feature+1]
            for j_feature in range(len(features_sorted)):
                j_start, j_end = cols_cum[j_feature], cols_cum[j_feature+1]
                new_r = np.max(r_matrix[i_start:i_end, j_start:j_end])
                final_r_matrix[i_feature, j_feature] = new_r
        return final_r_matrix

    def get_slice(self, key, rev_lookup_table=None):
        if rev_lookup_table is None:
            rev_lookup_table = self.rev_lookup_table
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError("Wrong dimension")
            rows = key[0]
            if isinstance(key[1], slice):
                srange = convert_slice(key[1], self.rev_lookup_table)
                cols = [i for i in range(*srange)]
            else:
                try:
                    int(key[1])
                    cols = [key[1]]
                except TypeError:
                    cols = key[1]
            col_ids = []
            for i_feature in cols:
                col_ids.extend(self.rev_lookup_table[i_feature]["col_ids"])
            return self.X[rows, col_ids]
        else:
            return self.X[key]

    @property
    def shape(self):
        return self.X.shape[0], len(self.rev_lookup_table)

    def kfold(self, y, k=8, rng=None, balance=True):
        if rng is None:
            rng = np.random.default_rng()

        X_folds = np.array_split(self.X, 8)
        y_folds = np.array_split(y, 8)

        def balance_fold(X, y):
            categories = np.unique(y)
            index_list = []
            for cat in categories:
                index_list.append(np.where(y == cat)[0])
            n_select = np.min([len(x) for x in index_list])
            select_list = []
            for cur_idx in index_list:
                if len(cur_idx) == n_select:
                    select_list.append(cur_idx)
                else:
                    add_idx = rng.choice(cur_idx, size=n_select, replace=False)
                    select_list.append(add_idx)
            select = np.sort(np.concatenate(select_list))
            return X[select], y[select]

        for i_fold in range(k):
            y_val = y_folds[i_fold]
            X_val = X_folds[i_fold]
            y_train = np.concatenate(y_folds[0:i_fold] + y_folds[i_fold+1:k])
            X_train = np.concatenate(X_folds[0:i_fold] + X_folds[i_fold+1:k])

            if balance:
                X_train, y_train = balance_fold(X_train, y_train)
                X_val, y_val = balance_fold(X_val, y_val)
            yield (FeatureMatrix(X_train, self.rev_lookup_table), y_train,
                   FeatureMatrix(X_val, self.rev_lookup_table), y_val)


def convert_slice(s, rev_lookup_table):
    if s.start is None:
        start = 0
    else:
        start = s.start
    if s.stop is None:
        stop = len(rev_lookup_table)
    else:
        stop = s.stop
    if s.step is None:
        step = 1
    else:
        step = s.step
    return start, stop, step


class FeatureMatrixView():
    def __init__(self, feature_matrix, subset):
        self._fm = feature_matrix
        self.rev_lookup_table = [feature_matrix.rev_lookup_table[i]
                                 for i in subset]

    def get_slice(self, key, rev_lookup_table=None):
        if rev_lookup_table is None:
            rev_lookup_table = self.rev_lookup_table
        return self._fm.get_slice(key, rev_lookup_table)

    def __getitem__(self, key):
        return self.get_slice(key)
