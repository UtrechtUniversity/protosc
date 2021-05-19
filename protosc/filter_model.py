import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.metrics import f1_score
from scipy import stats


def select_fold(y_folds, X_folds, i_val, balance=True):
    n_fold = len(y_folds)
    y_val = y_folds[i_val]
    X_val = X_folds[i_val]
    y_train = np.concatenate(y_folds[0:i_val] + y_folds[i_val+1:n_fold])
    X_train = np.concatenate(X_folds[0:i_val] + X_folds[i_val+1:n_fold])

    if not balance:
        return y_val, X_val, y_train, X_train

    train_one = np.where(y_train == 1)[0]
    train_zero = np.where(y_train == 0)[0]
    if len(train_one) >= len(train_zero):
        train_one = np.random.choice(train_one, size=len(train_zero),
                                     replace=False)
    else:
        train_zero = np.random.choice(train_zero, size=len(train_one),
                                      replace=False)
    selected_data = np.append(train_one, train_zero)
    np.random.shuffle(selected_data)
    return y_val, X_val, y_train[selected_data], X_train[selected_data]


def model_selected(y_training, X_training, y_val, X_val, selected_features,
                   kernel):
    """ Train an SVM on the train set while using the n selected features,
    crossvalidate on holdout """

    svclassifier = SVC(kernel=kernel)
    svclassifier.fit(X_training[:, selected_features], y_training)

    y_predict = svclassifier.predict(X_val[:, selected_features])

    outcome = {"Accuracy": round(accuracy_score(y_val, y_predict), 2),
               "Precision": round(precision_score(y_val, y_predict), 2),
               "Recall": round(recall_score(y_val, y_predict), 2),
               "F1_score": round(f1_score(y_val, y_predict), 2)}

#     print(classification_report(list(y_val), list(y_predict)))

    return outcome


def model_all(y_training, X_training, y_val, X_val, kernel):
    """ Train an SVM on the train set while using all features, crossvalidate
    on holdout """

    svclassifier = SVC(kernel=kernel)
    svclassifier.fit(X_training, y_training)

    y_predict = svclassifier.predict(X_val)

    outcome = {"Accuracy": round(accuracy_score(y_val, y_predict), 2),
               "Precision": round(precision_score(y_val, y_predict), 2),
               "Recall": round(recall_score(y_val, y_predict), 2),
               "F1_score": round(f1_score(y_val, y_predict), 2)}

#     print(classification_report(list(y_val), list(y_predict)))

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


def select_features(y_training, X_training, chisq_threshold=0.25):
    """Sort the chi-squares from high to low while keeping
    track of the original indices (feature_id)"""

    # Calculate chi-square using kruskall-wallis per feature
    X_chisquare = calc_chisquare(y_training, X_training)

    # Make a vector containing the new order of the original feature indices
    # when chi-square is sorted from high to low
    feature_id = np.argsort(-X_chisquare)

    # Sort the chi-squares from high to low
    chisquare_sorted = X_chisquare[feature_id]

#     print(chisquare_sorted)
    # Calculated the cumulative sum of the chi-sqaure vector
    cumsum = chisquare_sorted.cumsum()

    # Normalize to 1
#     stand = (cumsum - np.min(cumsum))/np.ptp(cumsum)

    # Select features needed to reach .25 (i.e., the number of features (n)
    # used for the filter selection)
#     selected_features = feature_id[stand <= 0.25]
    selected_features = feature_id[cumsum/cumsum[-1] <= chisq_threshold]
#     print(selected_features, feature_id[cumsum/cumsum[-1] <= 0.25])

    print(
        f'{selected_features.shape[0]} feature(s) used for the filter '
        f'selection')

    return selected_features


def filter_model(X, y, feature_id=None, n_fold=8):
    if feature_id is None:
        feature_id = np.arange(len(y))

    # Split data into 8 partitions: later use 1 partition as validating data,
    # other 7 as train data
    y_folds = np.array_split(y, 8)
    X_folds = np.array_split(X, 8)
#     feature_id_folds = np.array_split(feature_id, 8)

    # Balance classes

    # Train an SVM on the train set (i.e, 7 of the 8 X/y_trainings)
    # while using all features,
    # crossvalidate on holdout (i.e., 1 of the 8 X/y_trainings)
    output = []
    for i_val in range(8):
        # Set 1 partition as validating data, other 7 as train data
        y_val, X_val, y_train, X_train = select_fold(y_folds, X_folds, i_val)
#         y_val = y_folds[i_val]
#         X_val = X_folds[i_val]
#         y_train = np.concatenate(y_folds[0:i_val] + y_folds[i_val+1:8])
#         X_train = np.concatenate(X_folds[0:i_val] + X_folds[i_val+1:8])

        # Build the SVM model with specified kernel
        # ('linear', 'rbf', 'poly', 'sigmoid')
        model_output = model_all(
            y_train, X_train, y_val, X_val, kernel='linear')
        output.append(model_output)

    final = dict((k, [d[k] for d in output]) for k in output[0])

    # Train an SVM on the train set while using the selected features
    # (i.e., making up 25% of chisquare scores), crossvalidate on holdout
    output_sel = []

    for i_val in range(n_fold):
        # Set 1 partition as validating data, other 7 as train data
        y_val, X_val, y_train, X_train = select_fold(y_folds, X_folds, i_val)
        # Select the top n features needed to make .25 from
        selected_features = select_features(y_train, X_train)

        # Build the SVM model with specified kernel ('linear', 'rbf', 'poly',
        # 'sigmoid') using only selected features
        model_sel_output = model_selected(
            y_train, X_train, y_val, X_val, selected_features,
            kernel='linear')
        output_sel.append(model_sel_output)

    final_sel = dict((k, [d[k] for d in output_sel]) for k in output_sel[0])
    print(final)
    print(final_sel)
