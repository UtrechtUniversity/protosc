import numpy as np
from copy import deepcopy
from collections import defaultdict

from matplotlib import pyplot as plt


class FeatureMatrix():
    """Feature matrix which is aware of combined features."""
    def __init__(self, X, rev_lookup_table=None, plotters={}):
        """Initialize feature matrix.

        Arguments
        ---------
        X: np.ndarray[:, :]
            Base matrix that has the columns of all the channels.
        rev_lookup_table: list[dict]
            Each list item is a dictionary that effectively groups columns
            into features. One list item contains one feature as exposed to
            the outside world. Each dictionary contains the following items:
            pipeline: str
                An identifier that keeps track of which feature extraction
                pipeline the feature belongs to.
            sub_feature_id: int
                Feature identifier within features of the same pipeline.
            col_ids: list[int]
                A list of column ids (of X) that belong to the feature.
        plotters: dict[str, Plotter]
            Dictionary containing a plotting object for each of the pipelines
            for which such an object exists.
        """
        self.X = X
        if rev_lookup_table is None:
            rev_lookup_table = [
                {"pipeline": "matrix",
                 "sub_feature_id": i,
                 "col_ids": [i]} for i in range(X.shape[1])
            ]

        self.rev_lookup_table = rev_lookup_table
        self.plotters = plotters

    @classmethod
    def from_pipe_data(cls, data):
        """Create feature matrix from PipeComplex data."""
        n_row = len(data)

        # First setup the structure.
        rev_lookup_table = []
        n_col = 0
        plotters = {}
        for key, features in data[0].items():
            if isinstance(features, tuple):
                assert test_ref_frames(data, key)
                plotters[key] = features[1]
                features = features[0]
            if len(features.shape) == 1:
                n_channel = 1
            else:
                n_channel = features.shape[1]
            for sub_feature_id in range(features.shape[0]):
                rev_lookup_table.append({
                    "pipeline": key,
                    "sub_feature_id": sub_feature_id,
                    "col_ids": [i+n_col for i in range(n_channel)]
                })
                n_col += n_channel

        # Fill the matrix with data.
        X = np.zeros((n_row, n_col))
        for i_data, data_packet in enumerate(data):
            i_col = 0
            for features in data_packet.values():
                if isinstance(features, tuple):
                    features = features[0]
                if len(features.shape) == 1:
                    n_channel = 1
                else:
                    n_channel = features.shape[1]
                for sub_feature_id in range(features.shape[0]):
                    X[i_data, i_col: i_col+n_channel] = features[
                        sub_feature_id]
                    i_col += n_channel
        return cls(X, rev_lookup_table, plotters=plotters)

    @classmethod
    def from_matrix(cls, X):
        return cls(X)

    def copy(self):
        "Create a deepcopy of itself."
        X_copy = self.X.copy()
        table_copy = deepcopy(self.rev_lookup_table)
        return self.__class__(X_copy, rev_lookup_table=table_copy)

    def add_random_columns(self, n):
        """Add random columns to the feature matrix.

        Arguments
        ---------
        n: int
            Number of columns to add. Each column has a single channel.
        """
        shape = (self.shape[0], n)
        added_X = np.random.randn(*shape)
        old_columns = self.X.shape[1]
        old_features = self.shape[1]
        self.X = np.hstack((self.X, added_X))
        for i in range(n):
            self.rev_lookup_table.append({
                "pipeline": "random",
                "sub_feature_id": old_features+i,
                "col_ids": [old_columns+i],
            })

    def __getitem__(self, key):
        return self.get_slice(key)

    def __setitem__(self, key, val):
        return self.set_slice(key, val)

    def corrcoef(self, features_sorted):
        """Compute the correlation coefficients."""
        X_new = self[:, features_sorted]
        r_matrix = np.corrcoef(X_new, rowvar=False)
        cols_occ = [len(self.rev_lookup_table[x]["col_ids"])
                    for x in features_sorted]

        # Reduce the matrix to the proper number of features.
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
        """Get a slice of the internal matrix.

        Arguments
        ---------
        key: (tuple, int, slice)
            Indexing of the matrix. The numbers are given in feature ids.
        rev_lookup_table: dict
            Allow slicing with a different feature table.

        Returns
        -------
        X: np.ndarray[:, :]
            Returns an numpy array, which is a subset of the feature matrix.
            Information on channels is lost, so generally this should be
            called just before Machine Learning is applied to the matrix.
        """
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

    def set_slice(self, key, val, rev_lookup_table=None):
        """Set some part of the feature matrix.

        Since it you need to know the internal structure/channels to
        directly set the matrix to different values, this function is
        rarely used.
        """
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
            self.X[rows, col_ids] = val
        else:
            self.X[key] = val

    @property
    def shape(self):
        return self.X.shape[0], len(self.rev_lookup_table)

    @property
    def size(self):
        return self.shape[0]*self.shape[1]

    def kfold(self, y, k=8, rng=None, balance=True):
        """Generator to loop over k folds.

        Arguments
        ---------
        y: np.ndarray[int]
            Outcome of the samples (categorical or binary).
        k: int
            Number of folds.
        rng: np.random.rng
            Random number generator to determine the precise folds.
        balance: bool
            If true, then ensure that samples are balanced over class.

        Returns
        -------
        X_train: FeatureMatrix
            Training feature matrix
        y_train: np.ndarray[int]
            Training outcome.
        X_val: FeatureMatrix
            Validation feature matrix
        y_val: np.ndarray[int]
            Validation outcome
        """
        if rng is None:
            rng = np.random.default_rng()

        X_folds = np.array_split(self.X, k)
        y_folds = np.array_split(y, k)

        # Balance the folds.
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

    def plot(self, feature_id):
        """Plot the relevant features.

        If features are part of a feature extraction method that
        hasn't supplied a plotting function, then a warning is shown instead.

        Arguments
        ---------
        feature_id: (list[int], int)
            The feature ids that are relevant.
        """
        try:
            int(feature_id)
            feature_id = [feature_id]
        except TypeError:
            pass

        # Sort the feature ids into separate lists for each pipeline.
        split_fids = defaultdict(lambda: [])
        for fid in feature_id:
            sub_fid = self.rev_lookup_table[fid]["sub_feature_id"]
            pipe_id = self.rev_lookup_table[fid]["pipeline"]
            if pipe_id not in self.plotters:
                pipe_id = "noplot"
            split_fids[pipe_id].append(sub_fid)

        # Give a warning if one or more features cannot be plotted.
        n_no_plot = len(split_fids.pop("noplot", []))
        if n_no_plot > 0:
            print(f"{n_no_plot}features could not be plotted, because "
                  "functionality is not available.")

        # Create a plot for each of the pipelines.
        for pipe_id, sub_ids in split_fids.items():
            print(pipe_id)
            plt.figure(dpi=100)
            plt.axis("off")
            plotter = self.plotters[pipe_id]
            plotter.plot(sub_ids)


def convert_slice(s, rev_lookup_table):
    """Convert slice to (start, stop, step)."""
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

    def convert_negative(val):
        if val >= 0:
            return val
        else:
            return len(rev_lookup_table)+val

    start = convert_negative(start)
    stop = convert_negative(stop)
    return start, stop, step


def test_ref_frames(data, key):
    names = np.array([d[key][1].name for d in data])
    return np.all(names == names[0])

# class FeatureMatrixView():
#     def __init__(self, feature_matrix, subset):
#         self._fm = feature_matrix
#         self.rev_lookup_table = [feature_matrix.rev_lookup_table[i]
#                                  for i in subset]
#
#     def get_slice(self, key, rev_lookup_table=None):
#         if rev_lookup_table is None:
#             rev_lookup_table = self.rev_lookup_table
#         return self._fm.get_slice(key, rev_lookup_table)
#
#     def __getitem__(self, key):
#         return self.get_slice(key)
