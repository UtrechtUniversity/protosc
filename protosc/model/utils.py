import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy import stats
from scipy.special import betainc
from sklearn.preprocessing import StandardScaler

from protosc.feature_matrix import FeatureMatrix


def train_xvalidate(X_train, y_train, X_val, y_val, kernel="linear"):
    """Train an SVM on the train set while using the n selected features,
    crossvalidate on holdout"""

    svclassifier = SVC(kernel=kernel)
    scaler = StandardScaler().fit(X_train)
    svclassifier.fit(scaler.transform(X_train), y_train)

    y_predict = svclassifier.predict(scaler.transform(X_val))
    return accuracy_score(y_val, y_predict)


def compute_accuracy(cur_fold, selected_features):
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
    if len(selected_features) == 0:
        return 0
    X_train, y_train, X_val, y_val = cur_fold
    output = train_xvalidate(
        X_train[:, selected_features], y_train,
        X_val[:, selected_features], y_val)
    return output


def train_kfold_validate(X, y, features=None):
    if features is None:
        features = np.arange(X.shape[1])
    accuracy = []
    for cur_fold in X.kfold(y, k=8):
        X_train, y_train, X_val, y_val = cur_fold
        new_acc = train_xvalidate(X_train[:, features], y_train,
                                  X_val[:, features], y_val)
        accuracy.append(new_acc)
    return np.mean(accuracy)


def calc_chisquare(X_training, y_training):
    """Per feature, calculate chi-square using kruskall-wallis between
    two classes"""
    X_chisquare = []
    y_split = []
    cats = np.unique(y_training)
    for i_cat in cats:
        y_split.append(y_training == i_cat)

    # Estimate difference between classes per feature
    for feature in range(X_training.shape[1]):
        x = X_training[:, feature]
#         x1 = x[y_training == 0]
#         x2 = x[y_training == 1]
        if len(x.shape) > 1:
            kruskal_res = []
            for i_sub_feature in range(x.shape[1]):
                comp_vecs = [x[y_split[i_cat], i_sub_feature]
                             for i_cat in range(len(cats))]
                kruskal_res.append(
                    stats.kruskal(*comp_vecs)
                )
            new_chisquare = np.max(kruskal_res)
#             new_chisquare = np.max([
#                 stats.kruskal(x1[:, i], x2[:, i]) for i in range(x.shape[1])
#             ])
        else:
            comp_vecs = [x[y_split[i_cat]] for i_cat in range(len(cats))]
            new_chisquare = stats.kruskal(*comp_vecs).statistic
        X_chisquare.append(new_chisquare)

    X_chisquare = np.array(X_chisquare)

    return X_chisquare


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


# def fast_chisquare(X_training, y_training):
#     N = X_training.shape[0]
#     one_idx = np.where(y_training == 1)[0]
#     zero_idx = np.where(y_training == 0)[0]
#     N_one = len(one_idx)
#     N_zero = len(zero_idx)
#     reverse_order = np.empty(N, dtype=int)
#
#     chisq = np.empty(X_training.shape[1])
#     for i_feature in range(X_training.shape[1]):
#         order = np.argsort(X_training[:, i_feature])
#         reverse_order[order] = np.arange(N)
#         r_zero_sq = (np.mean(reverse_order[zero_idx])+1)**2
#         r_one_sq = (np.mean(reverse_order[one_idx])+1)**2
#         chisq[i_feature] = 12/(N*(N+1))*(
#             N_one*r_one_sq+N_zero*r_zero_sq) - 3*(N+1)
#     return chisq


def compute_pval(r, n_data):
    df = n_data - 2
    ts = r * r * (df / (1 - r * r))
    p = betainc(0.5 * df, 0.5, df / (df + ts))
    return p


def create_clusters(features_sorted, X):
    if isinstance(X, FeatureMatrix):
        r_matrix = X.corrcoef(features_sorted)
    else:
        r_matrix = np.corrcoef(X[:, features_sorted], rowvar=False)
    x_links, y_links = np.where(np.triu(r_matrix, 1)**2 >= 0.5)
    rvals = r_matrix[x_links, y_links]
    pvals = compute_pval(rvals, X.shape[0])
    significant_pvals = (pvals < 0.01)
    x_links = x_links[significant_pvals]
    y_links = y_links[significant_pvals]
    feature_selected = np.zeros(len(features_sorted), dtype=bool)

    if len(x_links) == 0:
        return [[x] for x in features_sorted]
    cur_src = x_links[0]
    cur_cluster = [features_sorted[x_links[0]]]
    all_clusters = []
    for i_link in range(len(x_links)):
        if (feature_selected[x_links[i_link]] or
                feature_selected[y_links[i_link]]):
            continue
        if x_links[i_link] != cur_src:
            feature_selected[cur_src] = True
            cur_src = x_links[i_link]
            all_clusters.append(cur_cluster)
            cur_cluster = [features_sorted[cur_src]]
        cur_cluster.append(features_sorted[y_links[i_link]])
        feature_selected[y_links[i_link]] = True
    all_clusters.append(cur_cluster)
    rest = [f for f in features_sorted
            if f not in np.concatenate(all_clusters)]
    for feat in rest:
        all_clusters.append([feat])
    return all_clusters


def select_features(X, y, chisq_threshold=0.25):  # , fast_chisq=False):
    """Sort the chi-squares from high to low while keeping
    track of the original indices (feature_id)"""

    # Calculate chi-square using kruskall-wallis per feature
    X_chisquare = calc_chisquare(X, y)

    # Make a vector containing the new order of the original feature indices
    # when chi-square is sorted from high to low
    features_sorted = np.argsort(-X_chisquare)

    # Remove lowest 5%
    features_sorted = features_sorted[:int(len(features_sorted)*0.95)]

    # Sort the chi-squares from high to low
    chisquare_sorted = X_chisquare[features_sorted]

    # Create clusters
    clusters = create_clusters(features_sorted, X)

    # Calculated the cumulative sum of the chi-square vector
    cumsum = chisquare_sorted.cumsum()

    # Select features needed to reach .25 of standardized cumsum
    # (i.e., the number of features (n) usef for filter)
    selected_features = features_sorted[:np.argmax(
        cumsum/cumsum[-1] >= chisq_threshold)+1]

    # Select clusters with n features
    final_selection = []
    for cluster in clusters:
        if len(final_selection) > len(selected_features):
            break
        final_selection.extend(cluster)

    return final_selection, clusters
