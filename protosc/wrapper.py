import numpy as np
from protosc.filter_model import train_xvalidate
from protosc.feature_matrix import FeatureMatrix


class Wrapper:
    def __init__(self, X, y, clusters, n_fold=8, search_space=0.15,
                 decrease=True, add_im=False, excl=False,
                 stop=4, n=1, fold_seed=None):
        """
        Args:
            X: np.array, FeatureMatrix
                Feature matrix to wrap.
            y: np.array
                Outcomes, categorical (0/1).
            clusters: np.array
                clusters of correlating features.
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
            n_fold: int
                number of folds (used for calculating accuracy).
            n: int,
                number of times you want to run the wrapper.
            fold_seed: int,
                seed
        """
        self.X = X
        self.y = y
        self.clusters = clusters
        self.n_fold = n_fold
        self.search_space = search_space
        self.decrease = decrease
        self.add_im = add_im
        self.excl = excl
        self.stop = stop
        self.n = n
        self.fold_seed = fold_seed

    def calc_accuracy(self, selection):
        """ Calculates the average accuracy score of the selected features over n_folds
        Args:
            selection: np.array,
                selected features used to calculate accuracy.
        Returns:
            accuracy: float,
                average accuracy score over n_folds.
        """
        # fold_seed = None
        fold_rng = np.random.default_rng(self.fold_seed)

        for cur_fold in self.X.kfold(self.y, k=self.n_fold, rng=fold_rng):
            accuracy = []
            X_train, y_train, X_val, y_val = cur_fold

            accuracy.append(train_xvalidate(
                X_train[:, selection], y_train, X_val[:, selection],
                y_val))
        return np.array(accuracy).mean()

    def __append_model(self, selected):
        """ Updates model with new selection of features
        Args:
            selected: list,
                indexes of clusters that increase accuracy.
        Returns:
            model: np.array,
                updated selection of features.
        """
        try:
            model = self.clusters[selected]
        except TypeError:
            model = np.array(self.clusters)[selected]
        return model

    def __exclude(self, selected, accuracy):
        """ Tries to increase accuracy of selected model by removing/replacing clusters
        Args:
            selected: list,
                selected cluster indexes used for model.
            accuracy: float,
                highest yielded accuracy from final model.
        Returns:
            if removal/replacement increased accuracy:
                function returns updated variables (i.e., new list of selected
                clusters & new highest accuracy).
        """
        print("""
        Trying to increase accuracy by removing/replacing clusters...
        """)
        exclude = []
        replace = {}
        for i in selected:
            # check if removing cluster increases accuracy
            rest = [x for x in selected if x != i and x not in exclude]
            selection = np.concatenate(np.array(self.clusters)[rest])
            accuracy_new = self.calc_accuracy(selection)
            if accuracy_new > accuracy:
                print(f"""
                Removed clusters {i}.
                Old accuracy: {accuracy}, new accuracy: {accuracy_new}.""")
                accuracy = accuracy_new
                exclude.append(i)
            else:
                # check if replacing cluster with new cluster
                # increases accuracy
                search = [x for x in range(len(self.clusters))
                          if x not in selected]
                search = search[:int(len(search)*self.search_space)]
                for j in search:
                    selection = np.append(selection, self.clusters[i])
                    accuracy_new = self.calc_accuracy(selection)
                    if accuracy_new > accuracy:
                        print(f"""
                        Replaced cluster {i} with {j}.
                        Old accuracy: {accuracy}, new accuracy: {accuracy_new}.
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

    def __matching(self, output):
        """ Find recurring clusters in wrapper output
        Args:
            output: dict,
                contains models, clusters, and accuracy scores
                of all wrapper runs.
        Returns:
            rec_clusters: list,
                cluster indeces that occur in each wrapper run.
        """
        all_runs = len(output['clusters'])
        all_clusters = [y for x in output['clusters'] for y in x]
        rec_clusters = []
        for x in set(all_clusters):
            if all_clusters.count(x) == all_runs:
                rec_clusters.append(x)
        return rec_clusters

    def wrapper(self):
        """ Determines which cluster of features yield the highest accuracy score
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
        # Define final output variable
        output = {'model': [], 'features': [], 'clusters': [], 'accuracy': []}

        # Repeat code n times
        for rounds in range(self.n):
            if self.n > 1:
                print(f'-- Round {rounds+1} of {self.n} --')

            # Define output variables
            selected = []
            model = []
            not_added = 0
            if not isinstance(self.X, FeatureMatrix):
                self.X = FeatureMatrix(self.X)

            # Define search order
            if self.decrease:
                cluster_order = range(len(self.clusters))
            else:
                cluster_order = reversed(range(len(self.clusters)))

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
                if cluster in selected:
                    continue

                # Set initial accuracy
                # if isinstance(model, list):
                if not model:
                    accuracy = 0

                # Determine search space
                rest = [x for x in range(len(self.clusters))
                        if x not in selected]
                rest = rest[:int(len(rest)*self.search_space)]

                # Look in search space for clusters that increase accuracy
                for i in rest:
                    # if isinstance(model, list):
                    if not model:
                        selection = self.clusters[i]
                    else:
                        selection = np.append(model, self.clusters[i])
                    accuracy_new = self.calc_accuracy(selection)

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
                            selected.append(i)
                            model = model + [self.clusters[sel_feature]]
                            print(f"""
                            added cluster {i}, new accuracy = {accuracy}""")

                # Only add cluster resulting in highest increase to model
                if self.add_im is False and added > 0:
                    selected.append(sel_feature)
                    model = model + [self.clusters[sel_feature]]
                    print(f"""
                    added cluster {sel_feature}, new accuracy = {accuracy}""")

                # If no clusters were added; increase 'not_added'
                # stopping value
                if added == 0:
                    print("""
                    nothing added""")
                    not_added += 1

            # Remove clusters
            if self.excl and len(selected) > 1:
                try:
                    selected, accuracy = self.__exclude(selected, accuracy)
                    model = np.array(self.clusters)[selected]
                except TypeError:
                    print("""
                    Removal/replacement of clusters did not increase accuracy.
                    """)

            # Add output per run to output dictionary
            output['model'].append(model)
            output['features'].append(np.concatenate(model))
            output['clusters'].append(selected)
            output['accuracy'].append(accuracy)

        if self.n > 1:
            output['recurring'] = self.__matching(output)

        return output
