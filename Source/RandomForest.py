import numpy as np
from joblib import Parallel, delayed


class CART:
    def __init__(self, f, max_depth=None):
        self.tree = None
        self.max_depth = max_depth
        self.f = f
        self.feature_freq = None

    def fit(self, X, y):
        self.feature_freq = {}
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_features = X.shape[1]
        n_labels = len(np.unique(y))

        if n_labels == 1:
            return {'prediction': y[0]}

        if depth == self.max_depth:
            return {'prediction': np.bincount(y).argmax()}

        # Each tree uses a random subspace (selection) of
        # features to split on at each node
        subset_features = np.random.choice(n_features, self.f, replace=False)

        all_the_same = 0
        for feature in subset_features:
            if len(np.unique(X[:, feature])) == 1:
                all_the_same += 1
        if all_the_same == len(subset_features):
            return {'prediction': np.bincount(y).argmax()}

        best_feature, best_threshold = self._best_criteria(X, y, subset_features)
        self.feature_freq[best_feature] = self.feature_freq.get(best_feature, 0) + 1

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
            sorted_thresholds = np.sort(thresholds)
            midpoints = (sorted_thresholds[:-1] + sorted_thresholds[1:]) / 2
            for threshold in midpoints:
                left_indices = X[:, feature_index] < threshold
                gini = self._gini_impurity(y[left_indices]) * np.sum(left_indices) / len(y) \
                       + self._gini_impurity(y[~left_indices]) * np.sum(~left_indices) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    @staticmethod
    def _gini_impurity(y):
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
    def __init__(self, nt=100, f=0, max_depth=None, n_jobs=-1):
        self.nt = nt
        self.max_depth = max_depth
        self.trees = []
        self.f = f
        self.feature_importances = None
        self.n_jobs = n_jobs

    def grow_tree(self, X, y):
        tree = CART(max_depth=self.max_depth, f=self.f)
        # The training set for each tree is
        # sampled (bootstrapping) from the original dataset
        bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
        bootstrap_X, bootstrap_y = X[bootstrap_indices], y[bootstrap_indices]
        tree.fit(bootstrap_X, bootstrap_y)
        return tree

    def fit(self, X, y):
        if self.f == 0:
            self.f = X.shape[1]

        self.trees = Parallel(n_jobs=self.n_jobs)(delayed(self.grow_tree)(X, y) for _ in range(self.nt))

        feature_counts = {}
        for tree in self.trees:
            for feature, freq in tree.feature_freq.items():
                feature_counts[feature] = feature_counts.get(feature, 0) + freq

        # Calculate feature importances
        total_trees = len(self.trees)
        self.feature_importances = {feature: count / total_trees for feature, count in feature_counts.items()}
        self.feature_importances = {k: v for k, v in
                                    sorted(self.feature_importances.items(), key=lambda item: item[1],
                                           reverse=True)}

    def predict(self, X):
        predictions = np.zeros((len(X), len(self.trees)), dtype=int)
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)
