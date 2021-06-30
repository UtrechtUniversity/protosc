import numpy as np
from protosc.filter_model import fast_chisquare, calc_chisquare, create_clusters, select_fold, train_xvalidate, select_features


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

    X_folds = np.array_split(X, 8)
    y_folds = np.array_split(y, 8)

    for i_val in range(n_fold):
        accuracy = []
        X_train, y_train, X_val, y_val = select_fold(X_folds, y_folds, i_val,
                                                     fold_rng)

        model_sel_output = train_xvalidate(X_train[:, selection],
                                           y_train,
                                           X_val[:, selection],
                                           y_val)
        accuracy.append(model_sel_output['Accuracy'])

    return np.array(accuracy).mean()


def wrapper(X, y, selected_clusters, n_fold=8):
    """ Determines the cluster of features yielding the highest accuracy scores
    Args:
        X: np.array, features
        y: np.array, categories (1/0)
        selected_clusters: np.array, selected cluster of features
        n_fold: int, number of folds
    Returns:
        model: np.array, selected cluster of features yielding the highest accuracy scores
    """
    selected = []
    model = []
    for cluster in range(len(selected_clusters)):
        if cluster in selected:
            next
        else:
            if isinstance(model, list):
                model = selected_clusters[cluster]
                try:
                    accuracy = calc_accuracy(X, y, model, n_fold)
                except ValueError:
                    accuracy = 0
                selected.append(cluster)
            elif isinstance(model, np.ndarray) and cluster not in selected:
                model = np.append(model, selected_clusters[cluster])
            print(f'model: {model}, selected {selected}')
            rest = [x for x in range(len(selected_clusters))
                    if x != cluster and x not in selected]
            print(f'rest = {rest}')
            for i in rest:
                selection = np.append(model, selected_clusters[i])
                print(f'selection = {selection}')
                accuracy_new = calc_accuracy(X, y, selection, n_fold)
                if accuracy_new > accuracy:
                    selected.append(i)
                    accuracy = accuracy_new
                    print(f'added {i}, new accuracy = {accuracy}')
                else:
                    next
            try:
                model = selected_clusters[selected]
            except TypeError:
                model = np.array(selected_clusters)[selected]

    return model, accuracy
