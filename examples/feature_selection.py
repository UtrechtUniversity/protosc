from pathlib import Path
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from scipy import stats

from examples.nimstim import execute
from protosc.preprocessing import GreyScale, ViolaJones, CutCircle
from protosc.feature_extraction import FourierFeatures
from protosc.io import ReadImage


def create_data(feature_list, y_all, files_all):
    """ Create dataframe with id, file path, category (O/1) and features per image """

    # Filter out exuberant (X): only compare open (O) vs. closed (C) mouths
    image_id = np.array(
        [image for image, select in enumerate(y_all != 2) if select])

    X = []
    for image in image_id:
        # for image in range(feature_list.shape[0]):
        data = [float(feature_list[image, feature].real)
                for feature in range(feature_list.shape[1])]
        X.append(data)

    X = np.array(X)

    y = y_all[y_all != 2]

    return X, y, image_id


def calc_chisquare(y_train, X_train):
    """ Calculate chi-square using kruskall-wallis per feature """

    X_chisquare = []

    for feature in range(X_train.shape[1]):
        x = X_train[:, feature]
        X_chisquare.append(stats.kruskal(x, y_train).statistic)

    X_chisquare = np.array(X_chisquare)

    return X_chisquare


def select_features(X_chisquare):
    """ Sort the chi-squares from high to low while keeping track of the original indices """

    # Make a vector containing the original feature indices in their new order (vector i)
    i = np.argsort(-X_chisquare)

    # Sort the chi-squares from high to low while keeping track of the original indices
    chisquare_sorted = X_chisquare[i]

    # Calculated the cumulative sum of the chi-sqaure vector
    cumsum = chisquare_sorted.cumsum()

    # normalize to 1
    stand = (cumsum - np.min(cumsum))/np.ptp(cumsum)

    # find how many features are needed to reach .25,
    # this will be the number of features (n) used for the filter selection
    selected_features = i[stand <= 0.25]

    return selected_features


def model_all(y_train, X_train, y_val, X_val, svclassifier=None):
    # Train an SVM on the train set while using all features, crossvalidate on holdout
    if not svclassifier:
        svclassifier = svm.SVC(kernel='linear')

    svclassifier.fit(list(X_train), list(y_train))

    y_pred = svclassifier.predict(list(X_val))
    outcome = classification_report(list(y_val), list(y_pred))

    return outcome


def model_selected(y_train, X_train, y_test, X_test,  svclassifier=None):
    # Per feature in the train data, calculate the chi-square using kruskall-wallis
    # (estimate difference between classes per feature basically)
    chi_statistics = calc_chisquare(y_train, X_train)

    # Select the top n features needed to make .25 from vector i
    selected_features = select_features(chi_statistics)

    # Train an SVM on the train set while using the selected features, crossvalidate on holdout
    if not svclassifier:
        svclassifier = svm.SVC(kernel='linear')

    svclassifier.fit(list(X_train[:, selected_features]), list(y_train))

    y_pred = svclassifier.predict(list(X_test[:, selected_features]))
    print(classification_report(list(y_test), list(y_pred)))

    return svclassifier


def main():

    # Define pipeline
    pipe1 = ReadImage() * ViolaJones(20) * CutCircle() * FourierFeatures()
    pipe2 = ReadImage() * GreyScale() * ViolaJones(20) * \
        CutCircle() * FourierFeatures()
    pipe_complex = pipe1 + pipe2

    # Set directory
    stim_data_dir = Path("..", "data", "Nimstim faces")

    # Create feature matrix
    feature_array, y_all, files_all = execute(
        pipe_complex, stim_data_dir, select='mouth', write=False)

    # Get overview
    print(f'Number of images: {len(files_all)}')
    print(f'Number of \'open mouth\' images: {sum(y_all == 1)}')
    print(f'Number of \'closed mouth\' images: {sum(y_all == 0)}')
    print(f'Number of \'other mouth\' images: {sum(y_all == 2)}')

    # Select one pipeline
    feature_list = feature_array[1]

    # Create dataframe with open vs. closed mouths
    X, y, image_id = create_data(feature_list, y_all, files_all)

    # Split data (features+labels) into train and test data + keep track of image_id
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, image_id, test_size=0.125)

    # Split data into 8 partitions: set 1 partition as test data, other 7 as train data
    y_trainings = np.array_split(y_train, 8)
    X_trainings = np.array_split(X_train, 8)
    id_trainings = np.array_split(id_train, 8)

    # Balance classes

    # Train an SVM on the train set while using all features, crossvalidate on holdout
    svclassifier_all = None
    for trainingset in range(7):
        svclassifier_all = model_all(y_trainings[trainingset],
                                     X_trainings[trainingset],
                                     y_test, X_test, svclassifier_all)

    # Train an SVM on the train set while using the selected features (i.e., making up 25% of chisquare scores), crossvalidate on holdout
    svclassifier_selected = None
    for trainingset in range(7):
        svclassifier_selected = model_selected(y_trainings[trainingset],
                                               X_trainings[trainingset],
                                               y_test, X_test, svclassifier_selected)

    # Redefine Train and Hold out data and repeat


if __name__ == "__main__":
    main()
