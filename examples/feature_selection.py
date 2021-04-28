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

    # Image ID's: Filter out exuberant (X), only compare open (0) vs. closed (2) mouths
    image_id = np.array([])
    for i in select:
        image_id = np.append(image_id.astype(int), np.where(y_all == i)[0])
    image_id = np.sort(image_id)

    # X: Filter out exuberant (X)
    X = []
    for image in image_id:
        data = [float(feature_list[image, feature].real)
                for feature in range(feature_list.shape[1])]
        X.append(data)
    X = np.array(X)

    # y: Filter out exuberant (X)
    y = y_all[y_all != 1]

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

    # Make a vector containing the original feature indices in their new order (vector feature_id)
    feature_id = np.argsort(-X_chisquare)

    # Sort the chi-squares from high to low while keeping track of the original indices
    chisquare_sorted = X_chisquare[feature_id]

    # Calculated the cumulative sum of the chi-sqaure vector
    cumsum = chisquare_sorted.cumsum()

    # normalize to 1
    stand = (cumsum - np.min(cumsum))/np.ptp(cumsum)

    # find how many features are needed to reach .25,
    # this will be the number of features (n) used for the filter selection
    selected_features = feature_id[stand <= 0.25]

    return selected_features


def model_all(y_training, X_training, y_val, X_val):
    # Train an SVM on the train set while using all features, crossvalidate on holdout
    svclassifier = svm.SVC(kernel='linear')
    svclassifier.fit(list(X_training), list(y_training))
    y_predict = svclassifier.predict(list(X_val))
    outcome = {"Accuracy": round(accuracy_score(y_val, y_predict), 2),
               "Precision": round(precision_score(y_val, y_predict), 2),
               "Recall": round(recall_score(y_val, y_predict), 2),
               "F1_score": round(f1_score(y_val, y_predict), 2)}
    print(classification_report(list(y_val), list(y_predict)))
    return outcome


def model_selected(y_training, X_training, y_val, X_val):
    # Per feature in the train data, calculate the chi-square using kruskall-wallis
    # (estimate difference between classes per feature basically)
    chi_statistics = calc_chisquare(y_training, X_training)

    # Select the top n features needed to make .25 from vector i
    selected_features = select_features(chi_statistics)

    # Train an SVM on the train set while using the selected features, crossvalidate on holdout
    svclassifier = svm.SVC(kernel='linear')

    svclassifier.fit(list(X_training[:, selected_features]), list(y_training))

    y_predict = svclassifier.predict(list(X_val[:, selected_features]))

    outcome = {"Accuracy": {round(accuracy_score(y_val, y_predict), 2)},
               "Precision": {round(precision_score(y_val, y_predict), 2)},
               "Recall": {round(recall_score(y_val, y_predict), 2)},
               "F1_score": {round(f1_score(y_val, y_predict), 2)}}

    print(classification_report(list(y_val), list(y_predict)))

    return outcome


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
    print(f'Number of \'open mouth\' images: {sum(y_all == 0)}')
    print(f'Number of \'exuberant mouth\' images: {sum(y_all == 1)}')
    print(f'Number of \'closed mouth\' images: {sum(y_all == 2)}')

    # Select one pipeline
    feature_list = feature_array[1]

    # Create dataframe with open vs. closed mouths
    X, y, image_id = create_data(feature_list, y_all, files_all, select=[0, 2])

    # Split data (features+labels) into train and test data + keep track of image_id
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, image_id, test_size=0.125)

    # Split data into 8 partitions: set 1 partition as test data, other 7 as train data
    y_trainings = np.array_split(y_train, 8)
    X_trainings = np.array_split(X_train, 8)
    id_trainings = np.array_split(id_train, 8)

    # Balance classes

    # Train an SVM on the train set while using all features, crossvalidate on holdout
    output = []
    for val in range(8):
        y_val = y_trainings[val]
        X_val = X_trainings[val]
        y_training = np.concatenate(y_trainings[0:val] + y_trainings[val+1:8])
        X_training = np.concatenate(X_trainings[0:val] + X_trainings[val+1:8])

        model_output = model_all(y_training, X_training, y_val, X_val)

        output.append(model_output)

    final = dict((k, [d[k] for d in output]) for k in output[0])

    # Train an SVM on the train set while using the selected features (i.e., making up 25% of chisquare scores), crossvalidate on holdout
    output_sel = []
    for val in range(8):
        y_val = y_trainings[val]
        X_val = X_trainings[val]
        y_training = np.concatenate(y_trainings[0:val] + y_trainings[val+1:8])
        X_training = np.concatenate(X_trainings[0:val] + X_trainings[val+1:8])

        model_sel_output = model_all(y_training, X_training, y_val, X_val)

        output_sel.append(model_sel_output)

    final_sel = dict((k, [d[k] for d in output]) for k in output_sel[0])

    # Redefine Train and Hold out data and repeat


if __name__ == "__main__":
    main()
