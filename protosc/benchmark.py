from protosc.wrapper import Wrapper
from protosc import filter_model
import numpy as np


def __run_models(X, y, n_fold=8, fold_seed=None):
    # Run filter model
    output_filter = filter_model(X, y, n_fold=n_fold, fold_seed=fold_seed)

    # Run slow wrapper
    slow = Wrapper(X, y, n_fold=n_fold, fold_seed=fold_seed)
    output_slow = slow.wrapper(n_jobs=-1)

    # Run fast wrapper
    fast = Wrapper(X, y, add_im=True, n_fold=n_fold, fold_seed=fold_seed)
    output_fast = fast.wrapper(n_jobs=-1)

    return output_filter, output_slow, output_fast


def __examine_filter(output_filter, n_fold=8):
    # Calculate average accuracy score
    accuracies = [a[1] for a in output_filter]
    av_accuracy = np.mean(accuracies)

    # Determine feature frequencies
    all_features = [f for feat in output_filter for f in feat[0]]
    fq_features = {}
    rec_features = []
    for x in set(all_features):
        fq_features[x] = all_features.count(x)
        # Find recurring features
        if all_features.count(x) == n_fold:
            rec_features.append(x)

    # Add all findings to one dicitonary
    results = {'Accuracy': av_accuracy, 'Recurring features': rec_features,
               'Feature frequencies': fq_features}
    return results


def __examine_wrapper(output_wrapper, n_fold=8):
    # Calculate average accuracy score
    av_accuracy = np.mean(output_wrapper['accuracy'])

    # Determine feature frequencies
    all_features = [f for feat in output_wrapper['features'] for f in feat]
    fq_features = {}
    for x in set(all_features):
        fq_features[x] = all_features.count(x)

    # Find recurring features
    rec_features = output_wrapper['recurring']

    # Add all findings to one dicitonary
    results = {'Accuracy': av_accuracy, 'Recurring features': rec_features,
               'Feature frequencies': fq_features}
    return results


def __compare_models(output_filter, output_slow, output_fast, n_fold=8):
    filter_model = __examine_filter(output_filter, n_fold=n_fold)
    wrapper_fast = __examine_wrapper(output_fast, n_fold=n_fold)
    wrapper_slow = __examine_wrapper(output_slow, n_fold=n_fold)
    dicts = [filter_model, wrapper_slow, wrapper_fast]
    overview = {k: [d[k] for d in dicts] for k in dicts[0]}
    return overview


def execute(X, y, fold_seed=None):
    output_filter, output_slow, output_fast = __run_models(
        X, y, fold_seed=fold_seed)
    overview = __compare_models(output_filter, output_slow, output_fast)
    return overview
