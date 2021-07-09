import numpy as np
from protosc.filter_model import train_xvalidate
from protosc.feature_matrix import FeatureMatrix


def calc_accuracy(X, y, selection, n_fold=8):
    """ Calculates the average accuracy score of the selected features over n_folds
    Args:
        X: np.array, features
        y: np.array, categories (1/0)
        selection: np.array, selected features
        n_fold: int, number of folds
    Returns:
        accuracy: float, average accuracy over n_folds
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


def __append_model(clusters, model, selected):
    """ Updates model
    Args:
        clusters: np.array, all clusters of correlating features
        model: np.array, currently selected features
        selected: list, indexes of clusters that increase accuracy
    Returns:
        model: np.array, updated selection of features
    """
    try:
        model = clusters[selected]
    except TypeError:
        model = np.array(clusters)[selected]
    return model


def wrapper(X, y, clusters, decrease=True, add_im=False, search_space=0.15,
            stop=4, n_fold=8):
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
        search_space: float
            percentage of clusters that will be used to select clusters from.
        stop: int
            max number of rounds where no clusters can be added,
            after which looping will stop
        n_fold: int
            number of folds (used for calculating accuracy)
    Returns:
        model: np.array, selected features yielding the highest accuracy scores
        selected: list, selected cluster indexes
        accuracy: float, highest yielded accuracy
    """
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
            print(f"No features were added in {stop} rounds. Stop searching.")
            break

        # If current cluster has already been selected, go to next
        if cluster in selected:
            continue

        # Select first cluster as model + calculate initial accuracy
        if isinstance(model, list):
            model = clusters[cluster]
            try:
                accuracy = calc_accuracy(X, y, model, n_fold)
            except ValueError:
                accuracy = 0
            selected.append(cluster)

        # Update model with nieuw cluster
        elif isinstance(model, np.ndarray) and cluster not in selected:
            model = np.append(model, clusters[cluster])
        print(f'selected clusters: {selected}')

        # Determine search space
        rest = [x for x in range(len(clusters))
                if x != cluster and x not in selected]
        rest = rest[:int(len(rest)*search_space)]

        # Look in search space for clusters that increase current accuracy
        # score
        for i in rest:
            selection = np.append(model, clusters[i])
            accuracy_new = calc_accuracy(X, y, selection, n_fold)

            # If accuracy is increased; save cluster and only add highest
            # increase to model
            if accuracy_new > accuracy:
                added += 1
                not_added = 0
                selected_feature = i
                accuracy = accuracy_new

                # If 'add immediately'; add said cluster to model and
                if add_im:
                    selected.append(i)
                    model = __append_model(
                        clusters, model, selected)
                    print(f'added cluster {i}, new accuracy = {accuracy}')
                    print(f'selected clusters: {selected}')

        # Add cluster resulting in highest increase to model
        if add_im is False and added > 0:
            selected.append(selected_feature)
            model = __append_model(clusters, model, selected)
            print(
                f'added cluster {selected_feature}, new accuracy = {accuracy}')

            # If no clusters were added; increase 'not_added' stopping value
            if added == 0:
                not_added += 1

    return model, selected, accuracy
