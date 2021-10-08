from copy import deepcopy

import numpy as np

from protosc.model.base import BaseFoldModel
from protosc.model.utils import select_features
from protosc.model.utils import compute_accuracy


class WrapperModel(BaseFoldModel):
    def __init__(self, n_fold=8, max_features=25, search_fraction=0.15,
                 reversed_clusters=True, greedy=False, exclusion_step=False,
                 max_nop_rounds=10):
        """
        Arguments:
        ----------
        n_fold: int
            number of folds
        max_features: int
            maximum number of features to use for SVM
        search_fraction: float
            percentage of clusters that will be used to select
            features from and add to model.
        reversed_clusters: boolean
            if True clusters are ranked from high to low chi-square scores,
            if False clusters are ranked from from low to high.
        greedy: boolean
            if True cluster is immediately added to the model if
                it increases the accuracy,
            if False it only adds the cluster with the highest
                accuracy increase.
        exclusion_step: boolean
            if True clusters are removed/replaced from the final model
                one by one to see if accuracy increases,
            if False this step is skipped.
        max_nop_rounds: int
            max number of rounds where no clusters are added to model,
            after which looping will stop.
        """
        self.n_fold = n_fold
        self.max_features = max_features
        self.search_fraction = search_fraction
        self.reversed_clusters = reversed_clusters
        self.greedy = greedy
        self.exclusion_step = exclusion_step
        self.max_nop_rounds = max_nop_rounds

    def copy(self):
        return self.__class__(
            max_features=self.max_features, n_fold=self.n_fold,
            search_fraction=self.search_fraction,
            reversed_clusters=self.reversed_clusters, greedy=self.greedy,
            exclusion_step=self.exclusion_step,
            max_nop_rounds=self.max_nop_rounds,
        )

    def _cluster_order(self, clusters):
        """ Define cluster order for search space.
        Args:
            clusters: np.array,
                clustered features.
        Returns:
            cluster_order: list,
                (reversed) list of cluster indeces.
        """
        if self.reversed_clusters:
            cluster_order = range(len(clusters))
        else:
            cluster_order = reversed(range(len(clusters)))
        return cluster_order

    def _remove_procedure(self, fold, selection, accuracy):
        """ Tries to increase accuracy of selected model by removing/replacing clusters
        Args:
            X_train/val: np.array, FeatureMatrix
                Training/validation set of feature matrix to wrap.
            y_train/val: np.array
                Training/validation set of categorical outcomes (0/1).
            clusters: np.array,
                clustered features.
            selected: list,
                selected cluster indexes used for model.
            accuracy: float,
                highest yielded accuracy from final model.
        Returns:
            if removal/replacement increased accuracy:
                function returns updated variables (i.e., new list of selected
                clusters & new highest accuracy).
        """
        exclude = []
        if len(selection) < 1:
            return selection, accuracy

        for i_cluster in deepcopy(selection.clusters):
            # check if removing cluster increases accuracy
            exclude_selection = selection - i_cluster
            accuracy_new = compute_accuracy(fold, exclude_selection.features)
            if accuracy_new > accuracy:
                accuracy = accuracy_new
                exclude.append(i_cluster)
                selection = exclude_selection
            else:
                # check if replacing cluster with new cluster
                # increases accuracy
                candidates = selection.search_space(
                    self.search_fraction,
                    exclude=exclude)
                new_selection, new_accuracy = self._add_clusters_max(
                    candidates, exclude_selection, accuracy, fold)
                diff_selection = new_selection - exclude_selection
                if len(diff_selection):
                    accuracy = new_accuracy
                    exclude.extend([i_cluster, diff_selection.clusters[0]])
                    selection = new_selection
        return selection, accuracy

    def _execute_fold(self, cur_fold):
        """ Runs wrapper one time.
        Args:
            X_train/val: np.array, FeatureMatrix
                Training/validation set of feature matrix to wrap.
            y_train/val: np.array
                Training/validation set of categorical outcomes (0/1).
        Returns:
            model: list,
                selected clustered features yielding the highest accuracy.
            features: np.array,
                selected features yielding the highest accuracy scores.
            accuracy: float,
                final (and highest) yielded accuracy of model.
        """
        # Define output variables
        n_not_added = 0
        accuracy = 0

        # Define search order
        X_train, y_train, _X_val, _y_val = cur_fold
        _, clusters = select_features(X_train, y_train)
        cluster_order = self._cluster_order(clusters)
        selection = ClusteredSelection(clusters)

        # Find clusters that increase accuracy
        for cluster in cluster_order:
            # If there were no features added in n rounds, stop searching
            if (n_not_added == self.max_nop_rounds
                    or len(selection.features) >= self.max_features):
                break
            # If current cluster has already been selected, go to next
            if cluster in selection.clusters:
                continue

            # Determine search space
            search_space = selection.search_space(self.search_fraction)

            # TODO: check if this is what is actually wanted!
            if self.greedy:
                new_selection, new_accuracy = self._add_clusters_direct(
                    search_space, selection, accuracy, cur_fold)
            else:
                new_selection, new_accuracy = self._add_clusters_max(
                    search_space, selection, accuracy, cur_fold)
            n_added = len(new_selection) - len(selection)
            if n_added:
                n_not_added = 0
            else:
                n_not_added += 1
            selection, accuracy = new_selection, new_accuracy

        if self.exclusion_step:
            # Remove clusters
            selection, accuracy = self._remove_procedure(
                cur_fold, selection, accuracy)
        return {
            "features": selection.features,
            "accuracy": accuracy,
        }

    def _add_clusters_direct(self, candidates, cur_selection,
                             cur_accuracy, fold):
        max_accuracy = cur_accuracy
        # Look in search space for clusters that increase accuracy
        for i_cluster in candidates:
            new_selection = cur_selection+i_cluster
            new_accuracy = compute_accuracy(fold, new_selection.features)

            # If accuracy is increased; update accuracy
            # and save cluster
            if new_accuracy > max_accuracy:
                cur_selection = new_selection
                max_accuracy = new_accuracy
        return cur_selection, max_accuracy

    def _add_clusters_max(self, candidates, cur_selection,
                          cur_accuracy, fold):
        max_accuracy = cur_accuracy
        i_max_accuracy = -1
        for i_cluster in candidates:
            new_selection = cur_selection + i_cluster
            new_accuracy = compute_accuracy(fold, new_selection.features)
            if new_accuracy > max_accuracy:
                i_max_accuracy = i_cluster
                max_accuracy = new_accuracy
        if max_accuracy > cur_accuracy:
            return cur_selection + i_max_accuracy, max_accuracy
        return cur_selection, max_accuracy


class ClusteredSelection():
    """Class for keeping track of clusters and features.

    Allows for adding and removing clusters and selecting a new
    search space according to their current clusters.
    """
    def __init__(self, all_clusters, clusters=[]):
        """Initialize class

        Arguments:
        ----------
        all_clusters: list[list[int]]
            Definition of all clusters, where each cluster is a list of indices
            that represents (combined) features from the feature matrix.
        clusters: list[int]
            Clusters that are selected at initialization. By default,
            no clusters are initialized.
        """
        self.all_clusters = all_clusters
        self.clusters = clusters
        if isinstance(self.all_clusters, np.ndarray):
            self.all_clusters = self.all_clusters.tolist()
        if isinstance(self.clusters, np.ndarray):
            self.clusters = self.clusters.tolist()

    @property
    def features(self):
        """Return the currently selected features."""
        cur_features = []
        for i_cluster in self.clusters:
            cur_features.extend(self.all_clusters[i_cluster])
        return cur_features

    def search_space(self, search_fraction, exclude=[]):
        """Return the search space given the current selection.

        Arguments:
        ----------
        search_fraction: float
            Only select this fraction from the available search space.
        exclude: list[int]
            Don't include these clusters in the search space

        Returns:
        --------
        search: list[int]
            List of clusters to be searched.
        """
        search = [x for x in range(len(self.all_clusters))
                  if x not in self.clusters and x not in exclude]
        # TODO: is this wanted?
        n_search = max(1, int(len(search)*search_fraction))
        search = search[:n_search]
        return search

    def copy(self):
        """Create a deepcopy of itself."""
        return self.__class__(self.all_clusters, deepcopy(self.clusters))

    def __add__(self, i_cluster):
        """Add a new cluster to the selection."""
        copy = self.copy()
        copy.clusters.append(i_cluster)
        return copy

    def __sub__(self, i_cluster):
        """Remove a cluster or a ClusteredSelection from the current one."""
        if isinstance(i_cluster, ClusteredSelection):
            diff_clusters = set(self.clusters) - set(i_cluster.clusters)
            copy = self.copy()
            copy.clusters = list(diff_clusters)
        else:
            copy = self.copy()
            copy.clusters.remove(i_cluster)
        return copy

    def __len__(self):
        return len(self.clusters)
