from protosc.wrapper import Wrapper
from protosc.filter_model import train_xvalidate, select_features
from protosc.feature_matrix import FeatureMatrix
from protosc.parallel import execute_parallel
import numpy as np
import random


def calc_accuracy(cur_fold, selected_features):
    """ Train an SVM on the train set while using the n selected features,
    crossvalidate on holdout (X/y_val)
    Args:
        cur_fold: tuple,
            contains X_train, y_train, X_val, y_val for current fold.
        selected_features: list,
            index of selected features used to train the SVM.
    Returns:
        output: int,
            returns accuracy of trained SVM.
    """
    X_train, y_train, X_val, y_val = cur_fold
    output = train_xvalidate(
        X_train[:, selected_features], y_train,
        X_val[:, selected_features], y_val)
    return output


def find_recurring(n_fold, output):
    """ Find features that occur in each run (i.e., n_fold).
    Args:
        output: dict,
            contains models, clusters, and accuracy scores
            of all wrapper runs.
    Returns:
        rec_features: list,
            features that occur in each wrapper run.
    """
    all_runs = n_fold
    all_features = [f for feat in output['features'] for f in feat]
    rec_features = []
    for x in set(all_features):
        if all_features.count(x) == all_runs:
            rec_features.append(x)
    return rec_features


def run_models(X, y,
               cur_fold,
               selected_features, clusters):
    """ Run every model for current fold 
    Args:
        X: np.array, FeatureMatrix
            Feature matrix to wrap.
        y: np.array
            Outcomes, categorical.
        cur_fold: tuple,
            contains X_train, y_train, X_val, y_val for current fold
        selected_features: list,
            index of selected features used to train the SVM.
        clusters: np.array,
            clustered features (based on correlation).
    Returns:
        output: dict,
            Per model:
                features: list with selected features.
                accuracy: final accuracy of selected features.
    """
    output = {}

    # Filtermodel
    filter_out = calc_accuracy(cur_fold, selected_features)
    output['filter'] = {'features': selected_features,
                        'accuracy': filter_out}

    # Wrapper fast
    fast = Wrapper(X, y, n=len(selected_features), stop=10, add_im=True)
    wrapper_out = fast._wrapper_once(cur_fold)
    output['fast_wrapper'] = {'features': wrapper_out[1],
                              'accuracy': wrapper_out[2]}

    # Wrapper slow
    slow = Wrapper(X, y, n=len(selected_features), stop=10, add_im=False)
    wrapper_out_slow = slow._wrapper_once(cur_fold)
    output['slow_wrapper'] = {'features': wrapper_out_slow[1],
                              'accuracy': wrapper_out_slow[2]}

    # Random
    random.shuffle(clusters)

    random_selection = []
    for cluster in clusters:
        if len(random_selection) >= len(selected_features):
            break
        random_selection.extend(cluster)
    random_out = calc_accuracy(cur_fold, random_selection)
    output['random'] = {'features': random_selection,
                        'accuracy': random_out}

    # Pseudo-random
    pseudo_selection = []
    for cluster in clusters:
        if len(pseudo_selection) >= len(selected_features):
            break
        for feat in cluster:
            if feat not in selected_features and \
                    feat not in wrapper_out[1]:
                pseudo_selection.append(feat)
    pseudo_out = calc_accuracy(cur_fold, pseudo_selection)
    output['pseudo'] = {'features': pseudo_selection,
                        'accuracy': pseudo_out}

    return output


def execute(X, y,
            n_fold=8, n_jobs=-1,
            fold_seed=1234, seed=1,
            feature_id=None,
            null_distribution=False,
            ):
    """ Run every model n_fold times parallel.
    Args:
        X: np.array, FeatureMatrix
            Feature matrix to wrap.
        y: np.array
            Outcomes, categorical.
        n_fold: int,
            number of folds you want to split the X and y data in.
        n_jobs: int,
            determines if you run the n_folds in parallel or not.
        fold_seed: int,
            seed for dividing folds.
        seed: int,
            seed random and numpy.
    Returns:
        final_result: dict,
            Per model:
                features: list with selected features.
                accuracy: final accuracy of selected features.
                recurring: list with recurring features in each fold.
    """
    if feature_id is None:
        feature_id = np.arange(len(y))

    if not isinstance(X, FeatureMatrix):
        X = FeatureMatrix(X)

    fold_rng = np.random.default_rng(fold_seed)

    np.random.seed(seed)
    random.seed(seed)

    results = []
    jobs = []
    for cur_fold in X.kfold(y, k=n_fold, rng=fold_rng):
        X_train, y_train, X_val, y_val = cur_fold
        if null_distribution:
            np.random.shuffle(y_train)
            cur_fold = X_train, y_train, X_val, y_val
        selected_features, clusters = select_features(X_train, y_train)
        jobs.append({
            "X": X,
            "y": y,
            "cur_fold": cur_fold,
            "selected_features": selected_features,
            "clusters": clusters
        })
        if n_jobs == 1 and n_fold != 1:
            results.append(run_models(
                X, y, cur_fold, selected_features, clusters))
            results

    if n_jobs != 1 and n_fold != 1:
        results = execute_parallel(jobs, run_models, n_jobs=n_jobs,
                                   progress_bar=True)

    final_result = {}
    for model in results[0].keys():
        dicts = [r[model] for r in results]
        final_result[model] = {k: [d[k] for d in dicts] for k in dicts[0]}
        final_result[model]['recurring'] = find_recurring(
            n_fold, final_result[model])

    return final_result
