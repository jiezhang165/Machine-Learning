"""
The goal of this practical lab is to implement a decision tree classifier

Pseudocode

DecisionTree:
if(stopping condition): return decision for this node
For each possible feature
For each possible split

Compute split points

Score the split using information gain

Take the feature and the split with the best score
Split the data points
Recurse on each subset

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Generate 200 2d feature points and their corresponding binary labels.
X = np.random.rand(200, 2)
y = np.zeros(200)
y[np.where(X[:, 0] < X[:, 1])] = 1

# Create color maps
cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#0000FF', '#FF0000'])

# Plot also the training points
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.show()


class Question:
    """A Question is used to partition a dataset.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if val < self.value:
            return True
        else:
            return False


q_test = Question(0, 0.4)
q_test.match(X[2, :])


def split(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def class_counts(rows):
    # Counts the number of each type of example in a dataset.
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def gini(rows):
    # Class_counts counts the number of each type of example in a dataset.
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    return impurity


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def optimal_split(rows):
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = split(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


# def decisionTree(rows):
#     gain, question = optimal_split(rows)
#     #stopping condition
#     if gain == 0:
#         # leaf node
#
#     true_rows, false_rows = split(rows, question)
#
#     # Recursively build the true branch.
#     true_branch = decisionTree(true_rows)
#
#     # Recursively build the false branch.
#     false_branch = decisionTree(false_rows)
#
#     # Return a Question node.
#     Question
