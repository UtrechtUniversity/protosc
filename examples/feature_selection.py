from examples.nimstim import NimStim
from pathlib import Path
from protosc.preprocessing import GreyScale, ViolaJones, CutCircle
from protosc.pipe_complex import PipeComplex
from protosc.feature_extraction import FourierFeatures
from protosc.io import ReadImage
grayscale = True
from sklearn.model_selection import train_test_split

# Define pipeline
pipe1 = ReadImage() * ViolaJones(20) * CutCircle() * FourierFeatures()
pipe2 = ReadImage() * GreyScale() * ViolaJones(20) * CutCircle() * FourierFeatures()
pipe_complex = pipe2

# Set directory
stim_data_dir = Path("..", "data", "Nimstim_faces")

# Create feature matrix
nimstim = NimStim(stim_data_dir)
feature_matrix, x, files = nimstim.execute(pipe_complex)
print(f'Number of files: {len(files)}')
print(f'Number of open images: {sum(x == 1)}')
print(f'Number of closed images: {sum(x == 0)}')
print(f'Number of other images: {sum(x == 2)}')

# Split data (features+labels) into 8 partitions

# Set 1 partition as hold out data, other 7 as train data

# Per feature in the train data, calculate the chi-square using kruskall-wallis (estimate difference between classes per feature basically)

# Sort the chi-squares from high to low while keeping track of the original indices

# Make a vector containing the original feature indices in their new order (vector i)

# Calculated the cumulative sum of the chi-sqaure vector (so the first value is the same as the first in the chi-square vector, the second is the sum of the first and the second value from the chi-square vector, the third value is the sum of the first 3 values etc)

# normalize to 1

# find how many features are needed to reach .25, this will be the number of features (n) used for the filter selection

# Define the filter selected by selecting the top n features from vector i

# Train an SVM on the train set while using all features, crossvalidate on holdout

# Train an SVM on the train set while using the selected features, crossvalidate on holdout

# Redefine Train and Hold out data and repeat