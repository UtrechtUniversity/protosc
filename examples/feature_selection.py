from pathlib import Path
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
from sklearn import svm
from scipy import stats

from examples.nimstim import execute
from protosc.preprocessing import GreyScale, ViolaJones, CutCircle
from protosc.feature_extraction import FourierFeatures
from protosc.io import ReadImage


def create_data(feature_list, y_all, files_all, select: list):
    """ Create X, y, and image_id arrays """

    # Image ID's: Filter data (only select the two 'select' classes)
    image_id_0 = np.where(y_all == select[0])[0]
    image_id_1 = np.where(y_all == select[1])[0]

    # y_all: make two categories binary
    y_all[image_id_0] = 0
    y_all[image_id_1] = 1

    # y: Filter data (only select the two 'select' classes)
    image_id = np.sort(np.append(image_id_0, image_id_1))
    y = y_all[image_id]

    # X: Filter data (only select the two 'select' classes)
    X = feature_list[image_id]

    return X, y, image_id


def calc_chisquare(y_training, X_training):
    """ Per feature, calculate chi-square using kruskall-wallis between two classes """

    X_chisquare = []

    # Estimate difference between classes per feature
    for feature in range(X_training.shape[1]):
        x = X_training[:, feature][:, 0]
        x1 = x[y_training == 0]
        x2 = x[y_training == 1]
        X_chisquare.append(stats.kruskal(x1, x2).statistic)

    X_chisquare = np.array(X_chisquare)

    return X_chisquare


def select_features(y_training, X_training):
    """ Sort the chi-squares from high to low while keeping track of the original indices (feature_id) """

    # Calculate chi-square using kruskall-wallis per feature
    X_chisquare = calc_chisquare(y_training, X_training)

    # Make a vector containing the new order of the original feature indices when chi-square is sorted from high to low
    feature_id = np.argsort(-X_chisquare)

    # Sort the chi-squares from high to low
    chisquare_sorted = X_chisquare[feature_id]

    # Calculated the cumulative sum of the chi-sqaure vector
    cumsum = chisquare_sorted.cumsum()

    # Normalize to 1
    stand = (cumsum - np.min(cumsum))/np.ptp(cumsum)

    # Select features needed to reach .25 (i.e., the number of features (n) used for the filter selection)
    selected_features = feature_id[stand <= 0.25]

    print(
        f'{selected_features.shape[0]} feature(s) used for the filter selection')

    return selected_features


def model_all(y_training, X_training, y_val, X_val, kernel):
    """ Train an SVM on the train set while using all features, crossvalidate on holdout """

    svclassifier = svm.SVC(kernel=kernel)
    svclassifier.fit(X_training, y_training)

    y_predict = svclassifier.predict(X_val[:, 0])

    outcome = {"Accuracy": round(accuracy_score(y_val, y_predict), 2),
               "Precision": round(precision_score(y_val, y_predict), 2),
               "Recall": round(recall_score(y_val, y_predict), 2),
               "F1_score": round(f1_score(y_val, y_predict), 2)}

    print(classification_report(list(y_val), list(y_predict)))

    return outcome


def model_selected(y_training, X_training, y_val, X_val, selected_features, kernel):
    """ Train an SVM on the train set while using the n selected features, crossvalidate on holdout """

    svclassifier = svm.SVC(kernel=kernel)
    svclassifier.fit(X_training[:, selected_features][:, 0], y_training)

    y_predict = svclassifier.predict(X_val[:, selected_features][:, 0])

    outcome = {"Accuracy": round(accuracy_score(y_val, y_predict), 2),
               "Precision": round(precision_score(y_val, y_predict), 2),
               "Recall": round(recall_score(y_val, y_predict), 2),
               "F1_score": round(f1_score(y_val, y_predict), 2)}

    print(classification_report(list(y_val), list(y_predict)))

    return outcome


def main():

    # Define pipeline
    pipe1 = ReadImage() * ViolaJones(20) * CutCircle() * FourierFeatures()
    pipe2 = ReadImage() * GreyScale() * ViolaJones(20) * \
        CutCircle() * FourierFeatures()
    pipe_complex = pipe1 + pipe2

    # Set directory to images
    stim_data_dir = Path("..", "data", "Nimstim faces")

    # Create feature matrix from images
    feature_array, y_all, files_all = execute(
        pipe_complex, stim_data_dir, select='mouth', write=False)

    # Select two most common classes
    classes = np.unique(y_all, return_counts=True)
    selection = classes[0][np.argpartition(classes[1], -2)[-2:]]

    # Select one pipeline
    feature_list = feature_array[1]

    # Create X and y with only the two most common classes
    X, y, image_id = create_data(
        feature_list, y_all, files_all, select=selection)

    # Split features (X) and labels (y) into train and test data + keep track of image_id
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, image_id, test_size=0.125)

    # Split data into 8 partitions: later use 1 partition as validating data, other 7 as train data
    y_trainings = np.array_split(y_train, 8)
    X_trainings = np.array_split(X_train, 8)
    id_trainings = np.array_split(id_train, 8)

    # Balance classes

    # Train an SVM on the train set (i.e, 7 of the 8 X/y_trainings) while using all features,
    # crossvalidate on holdout (i.e., 1 of the 8 X/y_trainings)
    output = []
    for val in range(8):
        # Set 1 partition as validating data, other 7 as train data
        y_val = y_trainings[val]
        X_val = X_trainings[val]
        y_training = np.concatenate(y_trainings[0:val] + y_trainings[val+1:8])
        X_training = np.concatenate(X_trainings[0:val] + X_trainings[val+1:8])

        # Build the SVM model with specified kernel ('linear', 'rbf', 'poly', 'sigmoid')
        model_output = model_all(
            y_training, X_training, y_val, X_val, kernel='linear')
        output.append(model_output)

    final = dict((k, [d[k] for d in output]) for k in output[0])

    # Train an SVM on the train set while using the selected features (i.e., making up 25% of chisquare scores), crossvalidate on holdout
    output_sel = []

    for val in range(8):
        # Set 1 partition as validating data, other 7 as train data
        y_val = y_trainings[val]
        X_val = X_trainings[val]
        y_training = np.concatenate(y_trainings[0:val] + y_trainings[val+1:8])
        X_training = np.concatenate(X_trainings[0:val] + X_trainings[val+1:8])

        # Select the top n features needed to make .25 from
        selected_features = select_features(y_training, X_training)

        # Build the SVM model with specified kernel ('linear', 'rbf', 'poly', 'sigmoid') using only selected features
        model_sel_output = model_selected(
            y_training, X_training, y_val, X_val, selected_features, kernel='linear')
        output_sel.append(model_sel_output)

    final_sel = dict((k, [d[k] for d in output_sel]) for k in output_sel[0])


if __name__ == "__main__":
    main()
