import numpy as np
from protosc.feature_matrix import FeatureMatrix


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

    final_biases = np.zeros(n_features)
    final_biases[selected_features] = biases
    ground_truth = {"selected_features": selected_features, "biases": final_biases}
    return FeatureMatrix(feature_matrix), y, ground_truth


def create_correlated_data(n_base_features=200, n_examples=500,
                           n_true_features=10,
                           n_feature_correlated=5,
                           min_dev=0.25, max_dev=0.5,
                           corr_frac=0.9):
    """Create mock data set.
    200 base features with 5 derived/correlated features each (total 1K).
    """
    n_one = n_examples//2
    n_zero = n_examples - n_one
    n_features = n_base_features*n_feature_correlated

    y = np.append(np.ones(n_one, dtype=int), np.zeros(n_zero, dtype=int))
    np.random.shuffle(y)
    ones = np.where(y == 1)[0]
    base_feature_matrix = np.random.randn(n_examples, n_base_features)
    feature_matrix = np.empty((n_examples,
                               n_base_features*n_feature_correlated))
    for i in range(n_feature_correlated):
        feature_matrix[:, i::n_feature_correlated] = (
            corr_frac*base_feature_matrix +
            (1-corr_frac)*np.random.randn(n_examples, n_base_features))
    biases = np.linspace(min_dev, max_dev, n_true_features)
    biases *= (-1)**np.arange(n_true_features)
    bias_values = np.zeros(n_features)
    clusters = (np.arange(n_features)/n_feature_correlated).astype(int)
    for base_feature_id in range(n_true_features):
        start = base_feature_id*n_feature_correlated
        end = (base_feature_id+1)*n_feature_correlated
        feature_matrix[ones, start:end] += biases[base_feature_id]
        bias_values[start:end] = biases[base_feature_id]

    selected_features = np.zeros(n_features, dtype=np.bool)
    selected_features[:n_true_features*n_feature_correlated] = True

    reorder = np.random.permutation(n_features)

    X = feature_matrix[:, reorder]
    selected_features = selected_features[reorder]
    selected_features = np.where(selected_features)[0]
    biases = bias_values[reorder]
    clusters = clusters[reorder]
    ground_truth = {"selected_features": selected_features,
                    "biases": biases,
                    "clusters": clusters}
    return FeatureMatrix(X), y, ground_truth


def create_categorical_data(n_features=500, n_examples=500,
                            n_true_features=25,
                            min_dev=0.25, max_dev=0.5,
                            n_categories=5):
    y = (n_categories*np.arange(n_examples)/n_examples).astype(int)
    split_y = []
    for cat in range(n_categories):
        split_y.append((y == cat).astype(int))
    feature_matrix = np.random.randn(n_examples, n_features)
    biases = np.zeros(n_features)
    biases[:n_true_features] = np.linspace(min_dev, max_dev, n_true_features)
    biases[:n_true_features] *= (-1)**np.arange(n_true_features)
    selected_features = np.zeros(n_features, dtype=np.bool)
    selected_features[:n_true_features] = 1

    for i_feature in range(n_features):
        cur_bias = biases[i_feature]
        if cur_bias == 0:
            continue
        fractions = np.random.rand(n_categories)
        fractions = (n_categories/2)*fractions/np.sum(fractions)
        for i_cat in range(n_categories):
            feature_matrix[split_y[i_cat], i_feature] += (
                cur_bias*fractions[i_cat])

    feature_reorder = np.random.permutation(n_features)
    example_reorder = np.random.permutation(n_examples)

    X = feature_matrix[:, feature_reorder]
    X = X[example_reorder, :]
    y = y[example_reorder]
    biases = biases[feature_reorder]
    selected_features = np.where(selected_features[feature_reorder])[0]
    ground_truth = {
        "selected_features": selected_features,
        "biases": biases,
    }
    return FeatureMatrix(X), y, ground_truth


def compare_results(selected_features, ground_truth):
    total_bias = np.sum(np.abs(ground_truth["biases"]))
    selected_bias = np.sum(np.abs(ground_truth["biases"][selected_features]))
    n_total_features = len(ground_truth["selected_features"])
    n_correct_selected = np.sum(ground_truth["biases"][selected_features] != 0)
    n_false_selected = np.sum(ground_truth["biases"][selected_features] == 0)
    print(f"Percentage of features correct: "
          f"{n_correct_selected}/{(n_correct_selected+n_false_selected)}")
    print(f"Percentage of features found: "
          f"{n_correct_selected}/{n_total_features}")
    print(f"Percentage of bias found: {selected_bias/total_bias}")
