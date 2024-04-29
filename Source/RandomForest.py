import numpy as np
import time
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if n_labels == 1:
            return {'prediction': y[0]}

        if depth == self.max_depth:
            return {'prediction': np.bincount(y).argmax()}

        feature_indices = np.random.choice(n_features, n_features, replace=False)
        best_feature, best_threshold = self._best_criteria(X, y, feature_indices)

        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices

        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature': best_feature,
                'threshold': best_threshold,
                'left': left_subtree,
                'right': right_subtree}

    def _best_criteria(self, X, y, feature_indices):
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] < threshold
                gini = self._gini_impurity(y[left_indices]) * np.sum(left_indices) / len(y) \
                       + self._gini_impurity(y[~left_indices]) * np.sum(~left_indices) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree):
        if 'prediction' in tree:
            return tree['prediction']

        feature_value = x[tree['feature']]
        if feature_value < tree['threshold']:
            return self._predict_tree(x, tree['left'])
        else:
            return self._predict_tree(x, tree['right'])

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth)
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            bootstrap_X, bootstrap_y = X[bootstrap_indices], y[bootstrap_indices]
            tree.fit(bootstrap_X, bootstrap_y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros((len(X), len(self.trees)))
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        return np.round(np.mean(predictions, axis=1)).astype(int)

def process_dataset(name: str, train_file: str, test_file: str):
    """
    This function processes a dataset for training and testing purposes.
    :param name: The name or identifier of the dataset.
    :param train_file: The filename of the training dataset in CSV format. It must be located in the '../Data/csv/'
    directory.
    :param test_file: The filename of the testing dataset in CSV format. It must be located in the '../Data/csv/'
    directory.
    """
    outfile = open(fr'../Data/output/{name}_output.txt', 'w')

    print('############', file=outfile)
    print(f'DATASET - {name.upper()}', file=outfile)
    print('############', file=outfile)

    data_train = pd.read_csv(fr'../Data/csv/{train_file}')
    data_test = pd.read_csv(fr'../Data/csv/{test_file}')

    X_train = data_train.drop('class', axis=1).to_numpy()
    y_train = data_train['class'].to_numpy()
    X_test = data_test.drop('class', axis=1).to_numpy()
    y_test = data_test['class'].to_numpy()

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.fit_transform(y_test)

    train_start = time.time()
    rf = RandomForest()
    rf.fit(X_train, y_train_encoded)
    train_end = time.time()

    print(f'Training time: {(train_end-train_start):.2f}s', file=outfile)

    y_predict = rf.predict(X_test)

    print(f'Accuracy: {100*accuracy_score(y_test_encoded, y_predict):.2f}%', file=outfile)

    outfile.close()

    outfile = open(fr'../Data/output/{name}_output.txt', 'r')
    print(outfile.read())
    outfile.close()
