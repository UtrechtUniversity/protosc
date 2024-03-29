from collections import defaultdict
import numpy as np


def final_selection(feature_accuracy, null_accuracy):
    """Perform the final selection depending on the xvalidated accuracy."""
    feature_results = defaultdict(lambda: [0, []])

    # Keep track of which folds are significant.
    sign_fold = set()
    null_percentile_99 = [np.quantile(x, 0.99) for x in null_accuracy]

    # Loop over the folds.
    for i_fold, res in enumerate(feature_accuracy):
        # res[0] is the list with included features.
        for feature_id in res["features"]:
            feature_results[feature_id][0] += res["accuracy"]
            feature_results[feature_id][1].append(i_fold)

        # res[1] is the accuracy with those features.
        if res["accuracy"] > null_percentile_99[i_fold]:
            sign_fold = sign_fold | set([i_fold])
    feature_results = dict(feature_results)

    signif_features = []

    # Loop over individual (combined) features.
    for feature_id, feature_res in feature_results.items():
        sum_res, occurences = feature_res
        if len(sign_fold.intersection(occurences)) == 0:
            continue
        res = []
        # Compute the 99 percentile values to compare it to.
        for i_res in range(100):
            res.append(np.sum(
                [null_accuracy[i_val][i_res] for i_val in occurences]))
        if np.max(res) < sum_res:
            signif_features.append(feature_id)

    return signif_features
