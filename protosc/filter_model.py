import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy import stats
from scipy.special import betainc
from protosc.feature_matrix import FeatureMatrix


def train_xvalidate(X_train, y_train, X_val, y_val, kernel="linear"):
    """Train an SVM on the train set while using the n selected features,
    crossvalidate on holdout"""

    svclassifier = SVC(kernel=kernel)
    svclassifier.fit(X_train, y_train)

    y_predict = svclassifier.predict(X_val)
    return accuracy_score(y_val, y_predict)


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

    # Estimate difference between classes per feature
    for feature in range(X_training.shape[1]):
        x = X_training[:, feature]
        x1 = x[y_training == 0]
        x2 = x[y_training == 1]
        if len(x.shape) > 1:
            new_chisquare = np.max([
                stats.kruskal(x1[:, i], x2[:, i]) for i in range(x.shape[1])
            ])
        else:
            new_chisquare = stats.kruskal(x1, x2).statistic
        X_chisquare.append(new_chisquare)

    X_chisquare = np.array(X_chisquare)

    return X_chisquare


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
    features_sorted = features_sorted

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
    selected_clusters = []
    for cluster in clusters:
        if len(selected_clusters) > len(selected_features):
            break
        selected_clusters.extend(cluster)

    return selected_clusters, clusters


def filter_model(X, y, feature_id=None, n_fold=8, fold_seed=None,
                 null_distribution=False):
    if feature_id is None:
        feature_id = np.arange(len(y))

    if not isinstance(X, FeatureMatrix):
        X = FeatureMatrix(X)

    fold_rng = np.random.default_rng(fold_seed)

    # Split data into 8 partitions: later use 1 partition as validating data,
    # other 7 as train data

    # Train an SVM on the train set while using the selected features
    # (i.e., making up 25% of chisquare scores), crossvalidate on holdout
    output_sel = []

    for cur_fold in X.kfold(y, k=n_fold, rng=fold_rng):
        X_train, y_train, X_val, y_val = cur_fold

        # Select the top n features needed to make .25
        if null_distribution:
            np.random.shuffle(y_train)

        selected_features = select_features(X_train, y_train)

        # Build the SVM model with specified kernel ('linear', 'rbf', 'poly',
        # 'sigmoid') using only selected features
        model_sel_output = train_xvalidate(
            X_train[:, selected_features], y_train,
            X_val[:, selected_features], y_val)
        output_sel.append((selected_features, model_sel_output))

    return output_sel
