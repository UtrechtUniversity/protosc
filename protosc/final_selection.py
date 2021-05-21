from collections import defaultdict
import numpy as np


def final_selection(feature_accuracy, null_accuracy):
    feature_results = defaultdict(lambda: [0, []])
    sign_fold = set()
    null_percentile_99 = [np.quantile(x, 0.99) for x in null_accuracy]
    for i_val, res in enumerate(feature_accuracy):
        for feature_id in res[0]:
            feature_results[feature_id][0] += res[1]
            feature_results[feature_id][1].append(i_val)
        if res[1] > null_percentile_99[i_val]:
            sign_fold = sign_fold | set([i_val])
    feature_results = dict(feature_results)

    signif_features = []

    for feature_id, feature_res in feature_results.items():
        sum_res, occurences = feature_res
        if len(sign_fold.intersection(occurences)) == 0:
            continue
        res = []
        for i_res in range(100):
            res.append(np.sum(
                [null_accuracy[i_val][i_res] for i_val in occurences]))
        if np.quantile(res, 0.99) < sum_res:
            signif_features.append(feature_id)

    return signif_features
