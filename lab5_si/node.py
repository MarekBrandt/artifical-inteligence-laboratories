import numpy as np


class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature_idx = None
        self.feature_value = None
        self.node_prediction = None

    def gini_best_score(self, y_sorted, possible_splits):
        best_gain = -np.inf
        best_idx = 0

        for i in possible_splits:
            if i == 0:
                continue
            gini_left_set = y_sorted[:i]
            gini_right_set = y_sorted[i:]

            left_total = len(gini_left_set)
            right_total = len(gini_right_set)
            total = len(y_sorted)

            negative_left = 0
            positive_left = 0
            for number in gini_left_set:
                if number == 1:
                    positive_left += 1
                else:
                    negative_left += 1

            negative_right = 0
            positive_right = 0

            for number in gini_right_set:
                if number == 1:
                    positive_right += 1
                else:
                    negative_right += 1

            gini_left = 1 - (negative_left / left_total) ** 2 - (positive_left / left_total) ** 2
            gini_right = 1 - (negative_right / right_total) ** 2 - (positive_right / right_total) ** 2
            gini_total = 1 - ((negative_right + negative_left )/ total) ** 2 -\
                         ((positive_right +  positive_left)/ total) ** 2

            gini_gain = gini_total - (left_total/total*gini_left+right_total/total*gini_right)
            if gini_gain > best_gain:
                best_idx = i
                best_gain = gini_gain

        return best_idx, best_gain

    def split_data(self, X, y, idx, val):
        left_mask = X[:, idx] < val
        return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])

    def find_possible_splits(self, data):
        possible_split_points = []
        for idx in range(data.shape[0] - 1):
            if data[idx] != data[idx + 1]:
                possible_split_points.append(idx)
        return possible_split_points

    def find_best_split(self, X, y):
        best_gain = -np.inf
        best_split = None

        for d in range(X.shape[1]):
            order = np.argsort(X[:, d])
            y_sorted = y[order]
            possible_splits = self.find_possible_splits(X[order, d])
            idx, value = self.gini_best_score(y_sorted, possible_splits)
            if value > best_gain:
                best_gain = value
                best_split = (d, [idx, idx + 1])

        if best_split is None:
            return None, None

        best_value = np.mean(X[best_split[1], best_split[0]])

        return best_split[0], best_value

    def predict(self, x):
        if self.feature_idx is None:
            return self.node_prediction
        if x[self.feature_idx] < self.feature_value:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def train(self, X, y):
        self.node_prediction = np.mean(y)
        if X.shape[0] == 1 or self.node_prediction == 0 or self.node_prediction == 1:
            return True

        self.feature_idx, self.feature_value = self.find_best_split(X, y)
        if self.feature_idx is None:
            return True

        (X_left, y_left), (X_right, y_right) = self.split_data(X, y, self.feature_idx, self.feature_value)

        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            self.feature_idx = None
            return True

        # create new nodes
        self.left_child, self.right_child = Node(), Node()
        self.left_child.train(X_left, y_left)
        self.right_child.train(X_right, y_right)
