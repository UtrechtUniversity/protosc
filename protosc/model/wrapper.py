from copy import deepcopy


from protosc.model.utils import train_xvalidate, select_features
from protosc.model.base import BaseModel


class Wrapper(BaseModel):
    def __init__(self, n=25, n_fold=8, search_space=0.15,
                 decrease=True, add_im=False, excl=False,
                 stop=5):
        """
        Args:
            X: np.array, FeatureMatrix
                Feature matrix to wrap.
            y: np.array
                Outcomes, categorical (0/1).
            n: int
                maximum number of features to use for SVM
            decrease: boolean
                if True clusters are ranked from high to low chi-square scores,
                if False clusters are ranked from from low to high.
            add_im: boolean
                if True cluster is immediately added to the model if
                    it increases the accuracy,
                if False it only adds the cluster with the highest
                    accuracy increase.
            excl: boolean
                if True clusters are removed/replaced from the final model
                    one by one to see if accuracy increases,
                if False this step is skipped.
            search_space: float
                percentage of clusters that will be used to select
                    features from and add to model.
            stop: int
                max number of rounds where no clusters are added to model,
                    after which looping will stop.
            n_fold: int,
                number of times you want to run the wrapper.
            fold_seed: int,
                seed
        """
        self.n = n
        self.n_fold = n_fold
        self.search_space = search_space
        self.decrease = decrease
        self.add_im = add_im
        self.excl = excl
        self.stop = stop

    def copy(self):
        return self.__class__(
            n=self.n, n_fold=self.n_fold,
            search_space=self.search_space, decrease=self.decrease,
            add_im=self.add_im, excl=self.excl, stop=self.stop,
        )

    def __cluster_order(self, clusters):
        """ Define cluster order for search space.
        Args:
            clusters: np.array,
                clustered features.
        Returns:
            cluster_order: list,
                (reversed) list of cluster indeces.
        """
        if self.decrease:
            cluster_order = range(len(clusters))
        else:
            cluster_order = reversed(range(len(clusters)))
        return cluster_order

    def __calc_accuracy(self, X_train, y_train, X_val, y_val, selection):
        """ Calculates the average accuracy score of the selected features over n_folds
        Args:
            X_train/val: np.array, FeatureMatrix
                Training/validation set of feature matrix to wrap.
            y_train/val: np.array
                Training/validation set of categorical outcomes (0/1).
            selection: np.array,
                selected features used to calculate accuracy.
        Returns:
            accuracy: float,
                average accuracy score over n_folds.
        """
        accuracy = train_xvalidate(
            X_train[:, selection.features], y_train,
            X_val[:, selection.features], y_val)
        return accuracy

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
        if not self.excl or len(selection) <= 1:
            return selection, accuracy

        for i_cluster in selection.clusters:
            # check if removing cluster increases accuracy
            exclude_selection = selection - i_cluster
            accuracy_new = self.__calc_accuracy(
                *fold, exclude_selection)
            if accuracy_new > accuracy:
                accuracy = accuracy_new
                exclude.append(i_cluster)
                selection = exclude_selection
            else:
                # check if replacing cluster with new cluster
                # increases accuracy
                candidates = selection.search_space(
                    self.search_space,
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
        cluster_order = self.__cluster_order(clusters)
        selection = ClusteredSelection(clusters)

        # Find clusters that increase accuracy
        for cluster in cluster_order:
            # If there were no features added in n rounds, stop searching
            if n_not_added == self.stop or len(selection.features) >= self.n:
                break
            # If current cluster has already been selected, go to next
            if cluster in selection.clusters:
                continue

            # Determine search space
            search_space = selection.search_space(self.search_space)

            # TODO: check if this is what is actually wanted!
            if self.add_im:
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
            new_accuracy = self.__calc_accuracy(*fold, new_selection)

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
            new_accuracy = self.__calc_accuracy(*fold, new_selection)
            if new_accuracy > max_accuracy:
                i_max_accuracy = i_cluster
                max_accuracy = new_accuracy
        if max_accuracy > cur_accuracy:
            return cur_selection + i_max_accuracy, max_accuracy
        return cur_selection, max_accuracy


def wrapper_exec(wrapper):
    cur_fold = wrapper.cur_fold
    return wrapper._wrapper_once(cur_fold)


class ClusteredSelection():
    def __init__(self, all_clusters, init_clusters=[]):
        self.all_clusters = all_clusters
        self.clusters = init_clusters

    @property
    def features(self):
        cur_features = []
        for i_cluster in self.clusters:
            cur_features.extend(self.all_clusters[i_cluster])
        return cur_features

    @property
    def clustered_features(self):
        cur_features = []
        for i_cluster in self.clusters:
            cur_features.append(self.all_clusters[i_cluster])
        return cur_features

    def search_space(self, search_fraction, exclude=[]):
        search = [x for x in range(len(self.all_clusters))
                  if x not in self.clusters and x not in exclude]
        # TODO: is this wanted?
        search = search[:int(len(search)*search_fraction)]
        return search

    def copy(self):
        return self.__class__(self.all_clusters, deepcopy(self.clusters))

    def __add__(self, i_cluster):
        copy = self.copy()
        copy.clusters.append(i_cluster)
        return copy

    def __sub__(self, i_cluster):
        if isinstance(i_cluster, ClusteredSelection):
            diff_clusters = set(self.clusters) - set(i_cluster)
            copy = self.copy()
            copy.clusters = diff_clusters
        else:
            copy = self.copy()
            copy.clusters.remove(i_cluster)
        return copy

    def __len__(self):
        return len(self.clusters)