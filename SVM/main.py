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
X[np.where(X[:, 0] < X[:, 1]), 0] -= 0.1
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

# Linear classifier
X1 = X[np.where(X[:, 0] > X[:, 1])]
X2 = X[np.where(X[:, 0] < X[:, 1])]
m1 = np.mean(X1, axis=0)
m2 = np.mean(X2, axis=0)
S1 = np.cov(X1.T)
S2 = np.cov(X2.T)
Sw = S1 + S2
print(np.linalg.inv(Sw))
w = np.dot(np.linalg.inv(Sw), (m1 - m2))
xp = np.linspace(0, 1, 100)
w0 = (np.dot(w.T, m1) + np.dot(w.T, m2)) / 2
yp = (-w[0] * xp + w0) / w[1]
plt.figure(2)
plt.plot(xp, yp, 'g')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.show()

# Logistic function
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

logreg = LogisticRegression(C=1e5)
# Create an instance of Logistic Regression Classifier and fit the data.
clf = logreg.fit(x_train, y_train)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = x_train[:, 0].min() - .5, x_train[:, 0].max() + .5  # limit x
y_min, y_max = x_train[:, 1].min() - .5, x_train[:, 1].max() + .5  # limit y
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

y_pred = clf.predict(x_test)
y_v = y_pred - y_test

score = 0

for i in range(len(y_v)):
    if y_v[i] == 0:
        score += 1
score = score / len(y_v)

print(score)

# clf.score(x_train, y_train, sample_weight=None)

# using the primal form
linear_svc = svm.SVC(kernel='linear')
linear_svc.fit(x_train, y_train)
linear_svc.predict(x_test)
print("Accurancy is: ", linear_svc.score(x_test, y_test))
