import numpy as np
from protosc.filter_model import train_xvalidate
from protosc.feature_matrix import FeatureMatrix


def calc_accuracy(X, y, selection, n_fold=8):
    """ Calculates the average accuracy score of the selected features over n_folds
    Args:
        X: np.array,
            features
        y: np.array,
            categories (1/0)
        selection: np.array,
            selected features
        n_fold: int,
            number of folds
    Returns:
        accuracy: float,
            average accuracy over n_folds
    """
    fold_seed = None
    fold_rng = np.random.default_rng(fold_seed)

    for cur_fold in X.kfold(y, k=n_fold, rng=fold_rng):
        accuracy = []
        X_train, y_train, X_val, y_val = cur_fold

        accuracy.append(train_xvalidate(
            X_train[:, selection], y_train, X_val[:, selection],
            y_val))
    return np.array(accuracy).mean()


def __append_model(clusters, selected):
    """ Updates model
    Args:
        clusters: np.array,
            all clusters of correlating features
        selected: list,
            indexes of clusters that increase accuracy
    Returns:
        model: np.array,
            updated selection of features
    """
    try:
        model = clusters[selected]
    except TypeError:
        model = np.array(clusters)[selected]
    return model


def __exclude(X, y, clusters, selected, accuracy, n_fold, search_space):
    """ Tries to increase accuracy of selected model by removing clusters
    Args:
        X: np.array, FeatureMatrix
            Feature matrix to wrap.
        y: np.array
            Outcomes, categorical.
        clusters: np.array
            clusters of correlating features.
        n_fold: int
            number of folds (used for calculating accuracy)
        selected: list, selected cluster indexes
        accuracy: float, highest yielded accuracy
    Returns:
        if removal increased accuracy: function returns updated variables
        (i.e., selected & accuracy)
    """
    print(
        "Trying to increase accuracy by removing/replacing clusters...")
    exclude = []
    replace = {}
    for i in selected:
        # check if removing cluster increases accuracy
        rest = [x for x in selected if x != i and x not in exclude]
        selection = np.concatenate(np.array(clusters)[rest])
        accuracy_new = calc_accuracy(X, y, selection, n_fold)
        if accuracy_new > accuracy:
            print(
                f'Removed clusters {i}. \
                    Old accuracy: {accuracy}, New accuracy: {accuracy_new}')
            accuracy = accuracy_new
            exclude.append(i)
        else:
            # check if replacing cluster with new cluster increases accuracy
            search = [x for x in range(len(clusters))
                      if x not in selected]
            search = search[:int(len(search)*search_space)]
            for j in search:
                selection = np.append(selection, clusters[i])
                accuracy_new = calc_accuracy(X, y, selection, n_fold)
                if accuracy_new > accuracy:
                    print(
                        f'Replaced cluster {i} with {j}. \
                            Old accuracy: {accuracy}, \
                            New accuracy: {accuracy_new}')
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


def __matching(output):
    """ Find recurring clusters in wrapper output of multiple runs
    Args:
        output: dict,
            contains models, clusters, and accuracy scores of all wrapper runs
    Returns:
        rec_clusters: list,
            all recurring clusters
    """
    all_runs = len(output['clusters'])
    all_clusters = [y for x in output['clusters'] for y in x]
    rec_clusters = []
    for x in set(all_clusters):
        if all_clusters.count(x) == all_runs:
            rec_clusters.append(x)
    return rec_clusters


def wrapper(X, y, clusters,
            decrease=True, add_im=False, excl=False,
            search_space=0.15, stop=4, n_fold=8, n=1):
    """ Determines the cluster of features yielding the highest accuracy scores
    Args:
        X: np.array, FeatureMatrix
            Feature matrix to wrap.
        y: np.array
            Outcomes, categorical.
        clusters: np.array
            clusters of correlating features.
        decrease: boolean
            if True clusters are ranked from high to low chi-square scores,
            if False from low to high.
        add_im: boolean
            if True clusters are immediately added to model if they increase
            the accuracy, if False it only adds the cluster with the highest
            accuracy increase.
        excl: boolean
            if True clusters are removed from the final model one by one to
            see if accuracy increases, if False this step is skipped.
        search_space: float
            percentage of clusters that will be used to select clusters from.
        stop: int
            max number of rounds where no clusters can be added,
            after which looping will stop
        n_fold: int
            number of folds (used for calculating accuracy)
        n: int,
            number of times you want to run the code
    Returns:
        output: dictionary,
            model: np.array, 
                selected features yielding the highest accuracy scores
            selected: list, 
                selected cluster indexes
            accuracy: float, 
                highest yielded accuracy
    """
    # Define final output variable
    output = {'model': [], 'features': [],
              'clusters': [], 'accuracy': [], 'recurring': []}

    # Repeat code n times
    for rounds in range(n):
        if n > 1:
            print(f'-- Round {rounds+1} of {n} --')

        # Define output variables
        selected = []
        model = []
        not_added = 0
        if not isinstance(X, FeatureMatrix):
            X = FeatureMatrix(X)

        # Define search order
        if decrease:
            cluster_order = range(len(clusters))
        else:
            cluster_order = reversed(range(len(clusters)))

        # Find clusters that increase accuracy
        for cluster in cluster_order:
            # If there were no features added in n rounds, stop searching
            added = 0
            if not_added == stop:
                print(
                    f"No features were added in {stop} rounds. \
                        Stop searching for new clusters.")
                break

            # If current cluster has already been selected, go to next
            if cluster in selected:
                continue

            # Set initial accuracy
            if isinstance(model, list):
                accuracy = 0

            # Determine search space
            rest = [x for x in range(len(clusters))
                    if x not in selected]
            rest = rest[:int(len(rest)*search_space)]

            # Look in search space for clusters that increase accuracy
            for i in rest:
                if isinstance(model, list):
                    selection = clusters[i]
                else:
                    selection = np.append(model, clusters[i])
                accuracy_new = calc_accuracy(X, y, selection, n_fold)

                # If accuracy is increased; update accuracy and save cluster
                if accuracy_new > accuracy:
                    added += 1
                    not_added = 0
                    selected_feature = i
                    accuracy = accuracy_new

                    # If 'add immediately'; add said cluster to model
                    # immediately continue with this model for adding
                    # new clusters
                    if add_im:
                        selected.append(i)
                        model = __append_model(clusters, selected)
                        print(f'added cluster {i}, new accuracy = {accuracy}')

            # Only add cluster resulting in highest increase to model
            if add_im is False and added > 0:
                selected.append(selected_feature)
                model = __append_model(clusters, selected)
                print(
                    f'added cluster {selected_feature}, \
                        new accuracy = {accuracy}')

            # If no clusters were added; increase 'not_added' stopping value
            if added == 0:
                print('nothing added')
                not_added += 1

        # Remove clusters
        if excl and len(selected) > 1:
            try:
                selected, accuracy = __exclude(
                    X, y, clusters, selected, accuracy, n_fold, search_space)
                model = np.array(clusters)[selected]
            except TypeError:
                print("Removal/replacement of clusters did \
                     not increase accuracy.")

        # Add output per run to output dictionary
        output['model'].append(model)
        output['features'].append(np.concatenate(model))
        output['clusters'].append(selected)
        output['accuracy'].append(accuracy)

    output['recurring'] = __matching(output)

    return output
