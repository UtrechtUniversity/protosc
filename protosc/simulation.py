import numpy as np


def create_simulation_data(n_features=400, n_examples=500, n_true_features=25,
                           min_dev=0.25, max_dev=0.5):
    """Create mock data set.
    400 features with two classes and 250 examples per class.
    image_id = array, image identifier (0-499)
    y = array, image classes (0/1)
    X = array, 400 features per image (normal distribution, mean 0)
    """

    n_one = n_examples//2
    n_zero = n_examples - n_one

    y = np.append(np.ones(n_one, dtype=int), np.zeros(n_zero, dtype=int))
    np.random.shuffle(y)
    ones = np.where(y == 1)[0]
    feature_matrix = np.random.randn(n_examples, n_features)
    biases = np.linspace(min_dev, max_dev, n_true_features)
    biases *= (-1)**np.arange(n_true_features)
    selected_features = np.random.choice(n_features, size=n_true_features,
                                         replace=False)
    for i_feature, feature_idx in enumerate(selected_features):
        feature_matrix[ones, feature_idx] += biases[i_feature]
    return feature_matrix, y, selected_features, biases
