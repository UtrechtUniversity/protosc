from protosc.wrapper import Wrapper
from protosc.filter_model import train_xvalidate, select_features
from protosc.feature_matrix import FeatureMatrix
from protosc.parallel import execute_parallel
import numpy as np
import random


def calc_accuracy(cur_fold, selected_features):
    X_train, y_train, X_val, y_val = cur_fold
    model_sel_output = train_xvalidate(
        X_train[:, selected_features], y_train,
        X_val[:, selected_features], y_val)
    return model_sel_output


def execute(X, y,
            feature_id=None,
            n_fold=8,
            fold_seed=1234, null_distribution=False, seed=1,
            n_jobs=-1):

    if feature_id is None:
        feature_id = np.arange(len(y))

    if not isinstance(X, FeatureMatrix):
        X = FeatureMatrix(X)

    fold_rng = np.random.default_rng(fold_seed)

    np.random.seed(seed)
    random.seed(seed)

    jobs = []
    for cur_fold in X.kfold(y, k=n_fold, rng=fold_rng):
        X_train, y_train, X_val, y_val = cur_fold
        if null_distribution:
            np.random.shuffle(y_train)
        selected_features, clusters = select_features(X_train, y_train)
        jobs.append({
            "X": X,
            "y": y,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "selected_features": selected_features,
            "clusters": clusters
        })

    results = execute_parallel(jobs, run_models, n_jobs=n_jobs,
                               progress_bar=True)

    final_result = {}
    for model in results[0].keys():
        dicts = [r[model] for r in results]
        final_result[model] = {k: [d[k] for d in dicts] for k in dicts[0]}

    return final_result


def run_models(X, y,
               X_train, y_train, X_val, y_val,
               selected_features, clusters):
    output = {}
    cur_fold = X_train, y_train, X_val, y_val

    # Filtermodel
    filter_out = calc_accuracy(cur_fold, selected_features)
    output['filter'] = {'features': selected_features,
                        'accuracy': filter_out}

    # Wrapper fast
    fast = Wrapper(X, y, n=len(selected_features), add_im=True)
    wrapper_out = fast._wrapper_once(X_train, y_train, X_val, y_val)
    output['fast_wrapper'] = {'features': wrapper_out[1],
                              'accuracy': wrapper_out[2]}

    # Wrapper slow
    slow = Wrapper(X, y, n=len(selected_features), add_im=False)
    wrapper_out_slow = slow._wrapper_once(X_train, y_train, X_val, y_val)
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
