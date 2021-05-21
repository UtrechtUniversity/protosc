import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.metrics import f1_score
from scipy import stats


def select_fold(y_folds, X_folds, i_val, rng, balance=True):
    n_fold = len(y_folds)
    y_val = y_folds[i_val]
    X_val = X_folds[i_val]
    y_train = np.concatenate(y_folds[0:i_val] + y_folds[i_val+1:n_fold])
    X_train = np.concatenate(X_folds[0:i_val] + X_folds[i_val+1:n_fold])

    if not balance:
        return y_val, X_val, y_train, X_train

    train_one = np.where(y_train == 1)[0]
    train_zero = np.where(y_train == 0)[0]
    if len(train_one) > len(train_zero):
        train_one = rng.choice(train_one, size=len(train_zero),
                               replace=False)
    elif len(train_one) < len(train_zero):
        train_zero = rng.choice(train_zero, size=len(train_one),
                                replace=False)
    selected_data = np.sort(np.append(train_one, train_zero))
    return y_val, X_val, y_train[selected_data], X_train[selected_data]


def model_selected(y_training, X_training, y_val, X_val, selected_features,
                   kernel):
    """ Train an SVM on the train set while using the n selected features,
    crossvalidate on holdout """

    svclassifier = SVC(kernel=kernel)
    svclassifier.fit(X_training[:, selected_features], y_training)

    y_predict = svclassifier.predict(X_val[:, selected_features])

    outcome = {"Accuracy": accuracy_score(y_val, y_predict),
               "Precision": precision_score(y_val, y_predict),
               "Recall": recall_score(y_val, y_predict),
               "F1_score": f1_score(y_val, y_predict)}

    return outcome


def model_all(y_training, X_training, y_val, X_val, kernel):
    """ Train an SVM on the train set while using all features, crossvalidate
    on holdout """

    svclassifier = SVC(kernel=kernel)
    svclassifier.fit(X_training, y_training)

    y_predict = svclassifier.predict(X_val)

    outcome = {"Accuracy": accuracy_score(y_val, y_predict),
               "Precision": precision_score(y_val, y_predict),
               "Recall": recall_score(y_val, y_predict),
               "F1_score": f1_score(y_val, y_predict)}

    return outcome


def calc_chisquare(y_training, X_training):
    """Per feature, calculate chi-square using kruskall-wallis between
    two classes"""

    X_chisquare = []

    # Estimate difference between classes per feature
    for feature in range(X_training.shape[1]):
        x = X_training[:, feature]
        x1 = x[y_training == 0]
        x2 = x[y_training == 1]
        X_chisquare.append(stats.kruskal(x1, x2).statistic)

    X_chisquare = np.array(X_chisquare)

    return X_chisquare


def create_clusters(features_sorted, X):
    """ Create clusters with features that correlate with each other """

    clusters = []

    # Starting with the highest ranking feature, check which features correlate and group those features together
    for feat in features_sorted:
        cluster = [feat]
        features_sorted = np.delete(
            features_sorted, np.where(features_sorted == feat))
        # to reduce size; only check first 25% of all other features (ranked on X_chisquare value, see select_features())
        other = features_sorted[:int(len(features_sorted)*0.25)]
        for rest in other:
            test = stats.pearsonr(X[:, feat], X[:, rest])
            if test[0] > 0.5 and test[1] < 0.05:
                cluster.append(rest)
                features_sorted = np.delete(
                    features_sorted, np.where(features_sorted == rest))
        clusters.append(cluster)

    return clusters


def select_features(y_training, X_training, chisq_threshold=0.25):
    """Sort the chi-squares from high to low while keeping
    track of the original indices (feature_id)"""

    # Calculate chi-square using kruskall-wallis per feature
    X_chisquare = calc_chisquare(y_training, X_training)

    # Make a vector containing the new order of the original feature indices
    # when chi-square is sorted from high to low
    features_sorted = np.argsort(-X_chisquare)

    # Remove lowest 5%
    features_sorted = features_sorted[:int(len(features_sorted)*0.95)]

    # Sort the chi-squares from high to low
    chisquare_sorted = X_chisquare[features_sorted]

    # Create clusters
    clusters = create_clusters(features_sorted, X_training)

    # Calculated the cumulative sum of the chi-sqaure vector
    cumsum = chisquare_sorted.cumsum()

    # Select features needed to reach .25 of standardized cumsum (i.e., the number of features (n) usef for filter)
    selected_features = features_sorted[:np.argmax(
        cumsum/cumsum[-1] >= 0.25)+1]

    # Select clusters with n features
    count = 0
    selected_clusters = []
    while count < len(selected_features):
        for i in range(len(clusters)):
            selected_clusters.extend(clusters[i])
            count += len(clusters[i])
            if count >= 10:
                break

    return selected_clusters


def filter_model(X, y, feature_id=None, n_fold=8, fold_seed=None,
                 null_distribution=False):
    if feature_id is None:
        feature_id = np.arange(len(y))

#     np.random.seed(seed)
    fold_rng = np.random.default_rng(fold_seed)
#     np.random.seed(seed)

    # Split data into 8 partitions: later use 1 partition as validating data,
    # other 7 as train data
    y_folds = np.array_split(y, 8)
    X_folds = np.array_split(X, 8)

    # Train an SVM on the train set while using the selected features
    # (i.e., making up 25% of chisquare scores), crossvalidate on holdout
    output_sel = []

    for i_val in range(n_fold):
        # Set 1 partition as validating data, other 7 as train data
        y_val, X_val, y_train, X_train = select_fold(y_folds, X_folds, i_val,
                                                     fold_rng)

        # Select the top n features needed to make .25
        if null_distribution:
            np.random.shuffle(y_train)

        selected_features = select_features(y_train, X_train)

        # Build the SVM model with specified kernel ('linear', 'rbf', 'poly',
        # 'sigmoid') using only selected features
        model_sel_output = model_selected(
            y_train, X_train, y_val, X_val, selected_features,
            kernel='linear')
        output_sel.append((selected_features, model_sel_output["Accuracy"]))

    return output_sel
