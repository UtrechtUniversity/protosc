from pathlib import Path
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from scipy import stats

from examples.nimstim import execute
from protosc.preprocessing import GreyScale, ViolaJones, CutCircle
from protosc.feature_extraction import FourierFeatures
from protosc.io import ReadImage

# Define pipeline
pipe1 = ReadImage() * ViolaJones(20) * CutCircle() * FourierFeatures()
pipe2 = ReadImage() * GreyScale() * ViolaJones(20) * CutCircle() * FourierFeatures()
pipe_complex = pipe1 + pipe2

# Set directory
stim_data_dir = Path("..", "data", "Nimstim faces")

# Create feature matrix
feature_array, x_all, files_all = execute(pipe_complex, stim_data_dir, select='mouth', write=False)

print(f'Number of images: {len(files_all)}')
print(f'Number of \'open mouth\' images: {sum(x_all == 1)}')
print(f'Number of \'closed mouth\' images: {sum(x_all == 0)}')
print(f'Number of \'other mouth\' images: {sum(x_all == 2)}')

# Select one pipeline
pipelines = list(feature_array.keys())
feature_list = feature_array[pipelines[1]]


def create_data(feature_array, x_all, files_all):
    """ Filter out exuberant (X): only compare open (O) vs. closed (C) mouths and create dataframe """" 
    
    # Filter out exuberant (X): only compare open (O) vs. closed (C) mouths
    files = files_all[x_all != 2]
    x = x_all[x_all != 2]
    feature_pipe = [feature_list[image] for image, select in enumerate(x_all != 2) if select]
    image_id = [image for image, select in enumerate(x_all != 2) if select]

    # Create dataframe that will be used for further analysis
    data = pd.DataFrame({'image_id': image_id, 'file_path': files, 'category': x, 'features': feature_pipe})
    return data

data = create_data(feature_array, x_all, files_all)

# Split data (features+labels) into train and test data
X_train, X_test, y_train, y_test = train_test_split(data['category'], data['features'], test_size=0.125)

# Split data into 8 partitions: set 1 partition as test data, other 7 as train data
y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7 = np.array_split(y_train, 7)
X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7 = np.array_split(X_train, 7)

# Balance classes


# Per feature in the train data, calculate the chi-square using kruskall-wallis
# (estimate difference between classes per feature basically)
def chisquare(y_train, X_train):

    y_chisquare = []
    features = range(len(y_train.to_list()[0]))

    for feature in features:
        y = [float(y_train[image][feature].real) for image in y_train.index]
        y_chisquare.append(stats.kruskal(X_train, y).statistic)

    return y_chisquare


# Sort the chi-squares from high to low while keeping track of the original indices
# Make a vector containing the original feature indices in their new order (vector i)
def sort_chisquare(y_chisquare):
    chisquare_sorted = pd.DataFrame({'chisquare': y_chisquare})
    chisquare_sorted['feature_id'] = chisquare_sorted.index

    chisquare_sorted = chisquare_sorted.sort_values(by='chisquare', ascending=False).reset_index(drop=True)

    return chisquare_sorted

# Calculated the cumulative sum of the chi-sqaure vector
# normalize to 1
def cumsum_chisquare(original):

    chisquare_sorted['cumsum'] = chisquare_sorted['chisquare'].cumsum()

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(chisquare_sorted)

    scaled = pd.DataFrame(x_scaled)[2]
    chisquare_sorted['scaled'] = scaled

    return chisquare_sorted


# find how many features are needed to reach .25,
# this will be the number of features (n) used for the filter selection
# Define the filter selected by selecting the top n features from vector i
def select(original):

    cutoff = chisquare_sorted[chisquare_sorted['scaled'] <= 0.25]

    return cutoff

# Train an SVM on the train set while using all features, crossvalidate on holdout
clf = svm.SVC()
clf.fit(X_train1, y_train1)

# Train an SVM on the train set while using the selected features, crossvalidate on holdout


# Redefine Train and Hold out data and repeat

chi_statistics = chisquare(y_train1, X_train1)
df_sorted = sort_chisquare(chi_statistics)
df_scaled = cumsum_chisquare(df_sorted)
df_selected = select(df_scaled)

