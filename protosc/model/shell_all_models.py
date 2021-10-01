import numpy as np

from protosc.model.utils import select_features, compute_accuracy
from protosc.model.wrapper import Wrapper
from protosc.feature_matrix import FeatureMatrix
from protosc.parallel import execute_parallel
from protosc.model.random import RandomModel
from protosc.model.pseudo_random import PseudoRandomModel
from protosc.model.final_selection import final_selection


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


def compute_null_accuracy(cur_fold, selected_features):
    X_train, y_train, X_val, y_val = cur_fold
    y_train_new = np.random.permutation(y_train)
    y_val_new = np.random.permutation(y_val)
    new_fold = (X_train, y_train_new, X_val, y_val_new)
    return compute_accuracy(new_fold, selected_features)


def compute_null_distribution(results, cur_fold, n_tot_results=100):
    null_distribution = []
    for i, res in enumerate(results.values()):
        selected_features = res["features"]
        n_compute = (n_tot_results-len(null_distribution))//(len(results)-i)
        for _ in range(n_compute):
            null_distribution.append(
                compute_null_accuracy(cur_fold, selected_features))
    return null_distribution


def run_models(cur_fold, selected_features, clusters):
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
    filter_accuracy = compute_accuracy(cur_fold, selected_features)
    output['filter'] = {'features': selected_features,
                        'accuracy': filter_accuracy}

    # Wrapper fast
    fast_wrapper = Wrapper(n=len(selected_features), stop=10, add_im=True)
    output['fast_wrapper'] = fast_wrapper._execute_fold(cur_fold)

    # Wrapper slow
    slow_wrapper = Wrapper(n=len(selected_features), stop=10, add_im=False)
    output['slow_wrapper'] = slow_wrapper._execute_fold(cur_fold)

    # Random
    output['random'] = RandomModel.execute_with_clusters(
        cur_fold, clusters, selected_features)

    # Pseudo random
    output['pseudo_random'] = PseudoRandomModel.execute_from_wrap_results(
        cur_fold, clusters, selected_features,
        output['fast_wrapper']['features'])

    return output, compute_null_distribution(output, cur_fold)


def execute_all_models(
        X, y,
        n_fold=8, n_jobs=-1,
        fold_seed=1234, seed=1,
        feature_id=None,
        null_distribution=False):
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

    results = []
    jobs = []
    for cur_fold in X.kfold(y, k=n_fold, rng=fold_rng):
        X_train, y_train, X_val, y_val = cur_fold
        if null_distribution:
            np.random.shuffle(y_train)
            cur_fold = X_train, y_train, X_val, y_val
        selected_features, clusters = select_features(X_train, y_train)
        jobs.append({
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
    null_dist = [res[1] for res in results]
    for model in list(results[0][0]):
        model_output = [res[0][model] for res in results]
        selection = final_selection(model_output, null_dist)
        final_result[model] = selection
#         dicts = [r[model] for r in results]
#         final_result[model] = {k: [d[k] for d in dicts] for k in dicts[0]}
#         final_result[model]['recurring'] = find_recurring(
#             n_fold, final_result[model])

    return final_result
