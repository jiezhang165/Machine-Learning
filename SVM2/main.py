# Given these data point, we aim at learning a predictor
# using the support vector machine method
# We start by a simple case of separable data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm

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

# Primal solution
# using the primal form
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
linear_svc = svm.SVC(kernel='linear')
linear_svc.fit(x_train, y_train)
linear_svc.predict(x_test)
print("Accurancy is: ", linear_svc.score(x_test, y_test))

# Plot the decision boudary
w = linear_svc.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(0, 1)
yy = a * xx - linear_svc.intercept_[0] / w[1]
plt.plot(xx, yy, 'g')

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.show()

# Dual solution
dual_svc = svm.LinearSVC(dual=True)
dual_svc.fit(x_train, y_train)
dual_svc.predict(x_test)
print("Accurancy is: ", dual_svc.score(x_test, y_test))

# Soft margin primal and dual solutions
X = np.random.rand(200, 2)
y = np.zeros(200)
y[np.where(X[:, 0] < X[:, 1])] = 1
X[np.where(X[:, 0] < X[:, 1]), 0] += 0.1

# Create color maps
cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#0000FF', '#FF0000'])

# Plot also the training points
plt.figure(2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.show()

# Primal solution
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
linear_svc = svm.SVC(kernel='linear')
linear_svc.fit(x_train, y_train)
linear_svc.predict(x_test)
print("Accurancy of linear primal solution is: ", linear_svc.score(x_test, y_test))

poly_svc = svm.SVC(kernel='poly', degree=4)
poly_svc.fit(x_train, y_train)
poly_svc.predict(x_test)
print("Accurancy of polynome primal solution is: ", poly_svc.score(x_test, y_test))

rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.fit(x_train, y_train)
rbf_svc.predict(x_test)
print("Accurancy of Gaussian primal polynome solution is: ", rbf_svc.score(x_test, y_test))

sigmoid_svc = svm.SVC(kernel='sigmoid')
sigmoid_svc.fit(x_train, y_train)
sigmoid_svc.predict(x_test)
print("Accurancy of sigmoid primal polynome solution is: ", sigmoid_svc.score(x_test, y_test))

# Dual solution
dual_svc = svm.LinearSVC(dual=True)
dual_svc.fit(x_train, y_train)
dual_svc.predict(x_test)
print("Accurancy of the dual solution is: ", dual_svc.score(x_test, y_test))

x_min, x_max = x_train[:, 0].min() - .5, x_train[:, 0].max() + .5  # limit x
y_min, y_max = x_train[:, 1].min() - .5, x_train[:, 1].max() + .5  # limit y
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  #
Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])  # predict class labels for each point
Z = Z.reshape(xx.shape)
plt.figure(3)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

for row in x_train:
    if row[0] < row[1]:
        plt.scatter(row[0], row[1], color='y', marker='o')
    else:
        plt.scatter(row[0], row[1], color='b', marker='+')

plt.show()

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
print(xx.shape)
