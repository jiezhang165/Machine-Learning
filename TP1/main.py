import numpy as np
import numpy.matlib
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# # generation of a group of points
# n_samples = 20
# X, y = datasets.make_regression(n_samples, 1, noise=10)
# plt.figure(1)
# plt.scatter(X, y)
# plt.show()
#
# '''Boundary line'''
# # solution1:analytical methode
# plt.figure(2)
# plt.title('Boundary line calculated by the analytical methode')
# plt.scatter(X, y)
# x0 = np.matlib.ones((20, 1))
# PHI = np.hstack((x0, X))
# theta = np.dot(np.dot(np.linalg.inv(np.dot(PHI.T, PHI)), PHI.T), y)
# t = theta[0, 0] + theta[0, 1] * X
# plt.plot(X, t, 'r')
# plt.show()
#
# # solution2:gradient descent
# y = np.array([y]).reshape(n_samples, 1)
# alpha = 0.001
#
#
# def error_function(PHI, theta, y):
#     """Error function J definition."""
#     diff = np.dot(PHI, theta) - y
#     return 1 / (2 * n_samples) * np.dot(diff.T, diff)
#
#
# def gradient_function(PHI, theta, y):
#     """Gradient function"""
#     diff = np.dot(PHI, theta) - y
#     return 1 / n_samples * np.dot(PHI.T, diff)
#
#
# def gradient_descent(PHI, y, alpha):
#     """Perform gradient descent"""
#     theta = np.array([1, 1]).reshape(2, 1)
#     gradient = gradient_function(PHI, theta, y)
#     while not np.all(np.absolute(gradient) <= 1e-5):
#         theta = theta - alpha * gradient
#         gradient = gradient_function(PHI, theta, y)
#     return theta
#
#
# optimal = gradient_descent(PHI, y, alpha)
# print(optimal)
# L = optimal[0, 0] + optimal[1, 0] * X
# plt.figure(3)
# plt.title('Boundary line calculated by the Gradient descent')
# plt.scatter(X, y)
# plt.plot(X, L, 'g')
# plt.show()

'''RANSAC algorithm'''

'''Linear classification'''

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target
print(X)
print(Y)

plt.figure(4)
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')  # First 50 samples
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')  # Middle 50 samples
plt.scatter(X[100:, 0], X[100:, 1], color='green', marker='+', label='Virginica')  # Last 50 samples
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend()
plt.show()

logreg = LogisticRegression(C=1e5)
# Create an instance of Logistic Regression Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5  # limit x
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5  # limit y
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  #
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])  # predict class labels for each point
# np.c_[xx.ravel(), yy.ravel()]， create a 2 rows matrix in which
# each column corresponds a point on the grid
'''A=[[1,2,3],
      [4,5,6]
      [7,8,9]]
    A.ravel()
    =[1,2,3,4,5,6,7,8,9]'''


# Put the result into a color plot
Z = Z.reshape(xx.shape)
print(Z)
plt.figure(5)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')  # First 50 samples
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')  # Middle 50 samples
# plt.scatter(X[100:, 0], X[100:, 1], color='green', marker='+', label='Virginica')  # Last 50 samples
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.show()

