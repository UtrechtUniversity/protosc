import numpy as np
from protosc.filter_model import train_xvalidate, select_features
from protosc.feature_matrix import FeatureMatrix
from protosc.parallel import execute_parallel


class Wrapper:
    def __init__(self, X, y, n_fold=8, search_space=0.15,
                 decrease=True, add_im=False, excl=False,
                 stop=4, fold_seed=None, cur_fold=None):
        """
        Args:
            X: np.array, FeatureMatrix
                Feature matrix to wrap.
            y: np.array
                Outcomes, categorical (0/1).
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
        self.X = X
        self.y = y
        self.n_fold = n_fold
        self.search_space = search_space
        self.decrease = decrease
        self.add_im = add_im
        self.excl = excl
        self.stop = stop
        self.fold_seed = fold_seed
        self.cur_fold = cur_fold

    def copy(self, cur_fold):
        return self.__class__(
            X=self.X, y=self.y, n_fold=self.n_fold,
            search_space=self.search_space, decrease=self.decrease,
            add_im=self.add_im, excl=self.excl, stop=self.stop,
            fold_seed=self.fold_seed,
            cur_fold=cur_fold
            )

    def __check_X(self):
        """ Ensure X is indeed FeatureMatrix.
        Returns:
            X = FeatureMatrix
        """
        if not isinstance(self.X, FeatureMatrix):
            self.X = FeatureMatrix(self.X)
        return self.X

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

    def __selection(self, clusters, model, i):
        """ Append model with selected cluster.
        Args:
            clusters: np.array,
                clustered features.
            model: list,
                current selection of clustered features.
            i: int,
                index of newly selected cluster.
        Returns:
            selection: np.array,
                new selection of clustered features.
        """
        if not model:
            selection = clusters[i]
        else:
            selection = np.append(model, clusters[i])
        return selection

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
            X_train[:, selection], y_train, X_val[:, selection],
            y_val)
        return accuracy

    def __update_model(self, selected, model, clusters, i, accuracy):
        """ Append selected with selected cluster.
        Args:
            selected: list,
                list of selected cluster indeces leading to increased accuracy
            model: list,
                current selection of clustered features.
            clusters: np.array,
                clustered features.
            i: int,
                index of newly selected cluster.
            accuracy: float,
                accuracy of new model.
        Returns:
            model: np.array,
                new selection of clustered features.
            selected: list,
                new selection of cluster indeces.
        """
        selected.append(i)
        model = model + [clusters[i]]
        print(f"""
        added cluster {i}, new accuracy = {accuracy}""")
        return selected, model

    def __exclude(self, X_train, y_train, X_val, y_val,
                  selected, clusters, accuracy):
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
        replace = {}
        if self.excl and len(selected) > 1:
            print("""
            Trying to increase accuracy by removing/replacing clusters...
            """)
            for i in selected:
                # check if removing cluster increases accuracy
                rest = [x for x in selected if x != i and x not in exclude]
                selection = np.concatenate(clusters)
                selection = selection[rest]
                accuracy_new = self.__calc_accuracy(
                    X_train, y_train, X_val, y_val, selection)
                if accuracy_new > accuracy:
                    print(f"""
                    Removed clusters {i}.
                    Old accuracy: {accuracy}, new accuracy: {accuracy_new}.""")
                    accuracy = accuracy_new
                    exclude.append(i)
                else:
                    # check if replacing cluster with new cluster
                    # increases accuracy
                    search = [x for x in range(len(clusters))
                              if x not in selected]
                    search = search[:int(len(search)*self.search_space)]
                    for j in search:
                        selection = np.append(selection, clusters[i])
                        accuracy_new = self.__calc_accuracy(
                            X_train, y_train, X_val, y_val, selection)
                        if accuracy_new > accuracy:
                            print(f"""
                            Replaced cluster {i} with {j}.
                            Old accuracy: {accuracy}, new: {accuracy_new}.
                            """)
                            accuracy = accuracy_new
                            replace.update({i: j})
        if exclude:
            selected = [x for x in selected if x not in exclude]
            return selected, accuracy_new
        if replace:
            for x in range(len(selected)):
                if selected[x] in replace:
                    selected[x] = replace[selected[x]]
            return selected, accuracy_new

    def __empty_round(self, added, not_added):
        """ Update number of round no new clusters were added.
        Args:
            added: int,
                number of clusters added in that specific round.
            not_added: int,
                total rounds no new clusters were added.
        Returns:
            updated not_added
        """
        if added == 0:
            print("nothing added")
            not_added += 1
        return not_added

    def __matching(self, output):
        """ Find features that occur in each run (i.e., n_fold).
        Args:
            output: dict,
                contains models, clusters, and accuracy scores
                of all wrapper runs.
        Returns:
            rec_features: list,
                features that occur in each wrapper run.
        """
        all_runs = self.n_fold
        all_features = [f for feat in output['features'] for f in feat]
        rec_features = []
        for x in set(all_features):
            if all_features.count(x) == all_runs:
                rec_features.append(x)
        return rec_features

    def _wrapper_parallel(self, n_jobs=-1):
        """ Runs wrapper n_rounds times in parallel
        Args:
            n_jobs: int,
                number of jobs
        Returns:
            output: dictionary,
                model: np.array,
                    selected clustered features yielding the highest accuracy.
                features: np.array,
                    selected features yielding the highest accuracy scores.
                accuracy: float,
                    final (and highest) yielded accuracy of model.
        """
        output = {}
        fold_rng = np.random.default_rng(self.fold_seed)
        jobs = [{"wrapper": self.copy(cur_fold)} for cur_fold in self.X.kfold(
            self.y, k=self.n_fold, rng=fold_rng)]

        results = execute_parallel(jobs, wrapper_exec, n_jobs=n_jobs)

        output['model'] = [r[0] for r in results]
        output['features'] = [r[1] for r in results]
        output['accuracy'] = [r[2] for r in results]

        return output

    def _wrapper_once(self, X_train, y_train, X_val, y_val):
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
            clusters: list,
                indeces of selected clusters.
            accuracy: float,
                final (and highest) yielded accuracy of model.
        """
        # Define output variables
        selected = []
        model = []
        not_added = 0
        accuracy = 0

        # Define search order
        _, clusters = select_features(X_train, y_train)
        cluster_order = self.__cluster_order(clusters)

        # Find clusters that increase accuracy
        for cluster in cluster_order:
            # If there were no features added in n rounds, stop searching
            added = 0
            if not_added == self.stop:
                print(f"""
                No features were added in {self.stop} rounds.
                Stop searching for new clusters.""")
                break
            # If current cluster has already been selected, go to next
            elif cluster in selected:
                continue

            # Determine search space
            rest = [x for x in cluster_order
                    if x not in selected]
            rest = rest[:int(len(rest)*self.search_space)]

            # Look in search space for clusters that increase accuracy
            for i in rest:
                selection = self.__selection(clusters, model, i)
                accuracy_new = self.__calc_accuracy(
                    X_train, y_train, X_val, y_val, selection)

                # If accuracy is increased; update accuracy
                # and save cluster
                if accuracy_new > accuracy:
                    added += 1
                    not_added = 0
                    sel_feature = i
                    accuracy = accuracy_new

                    # If 'add immediately'; add said cluster to model
                    # immediately continue with this model for adding
                    # new clusters
                    if self.add_im:
                        selected, model = self.__update_model(selected, model,
                                                              clusters, i,
                                                              accuracy)

            # Only add cluster resulting in highest increase to model
            if self.add_im is False and added > 0:
                selected, model = self.__update_model(selected, model,
                                                      clusters, sel_feature,
                                                      accuracy)

            # If no clusters were added; increase 'not_added'
            # stopping value
            not_added = self.__empty_round(added, not_added)

        # Remove clusters
        try:
            selected, accuracy = self.__exclude(X_train, y_train, X_val, y_val,
                                                selected, clusters, accuracy)
            model = np.array(clusters)[selected]
        except TypeError:
            pass

        return model, np.concatenate(model), accuracy

    def wrapper(self, n_jobs=1):
        """ Determines which cluster of features yield the highest accuracy score.
        Args:
            n_jobs: int,
                number of jobs.
        Returns:
            output: dictionary,
                model: np.array,
                    selected clustered features yielding the highest accuracy.
                features: np.array,
                    selected features yielding the highest accuracy scores.
                clusters: list,
                    indeces of selected clusters.
                accuracy: float,
                    final (and highest) yielded accuracy of model.
                recurring: list,
                    if multiple wrapper runs: cluster indeces that occur in
                        each wrapper run.
        """
        self.X = self.__check_X()
        # Runs wrapper for n_fold runs parallel:
        if n_jobs != 1 and self.n_fold != 1:
            output = self._wrapper_parallel(n_jobs)
        # Runs wrapper for n_fold times:
        else:
            output = {'model': [], 'features': [], 'accuracy': []}
            fold_rng = np.random.default_rng(self.fold_seed)
            for cur_fold in self.X.kfold(self.y, k=self.n_fold, rng=fold_rng):
                X_train, y_train, X_val, y_val = cur_fold
                model, features, accuracy = self._wrapper_once(
                    X_train, y_train, X_val, y_val)
                output['model'].append(model)
                output['features'].append(features)
                output['accuracy'].append(accuracy)

        if self.n_fold > 1:
            output['recurring'] = self.__matching(output)

        return output


def wrapper_exec(wrapper):
    X_train, y_train, X_val, y_val = wrapper.cur_fold
    return wrapper._wrapper_once(X_train, y_train, X_val, y_val)
