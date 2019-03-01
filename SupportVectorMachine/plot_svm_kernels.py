"""
Three different types of SVM-Kernels are displayed below.
The polynomial and RBF are especially useful when the
data-points are not linearly separable.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Our dataset and targets
X = np.array([[.4, -.7],
              [-1.5, -1],
              [-1.4, -.9],
              [-1.3, -1.2],
              [-1.1, -.2],
              [-1.2, -.4],
              [-.5, 1.2],
              [-1.5, 2.1],
              [1, 1],
              [1.3, .8],
              [1.2, .5],
              [.2, -2],
              [.5, -2.4],
              [.2, -2.3],
              [0, -2.7],
              [1.3, 2.1]])
Y = [0] * 8 + [1] * 8

plt.figure(1, figsize=(16, 5))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

# fit the model
for kernel, ax in (('linear', ax1), ('poly', ax2), ('rbf', ax3)):
    clf = svm.SVC(kernel=kernel, gamma=2)
    clf.fit(X, Y)

    plt.sca(ax)
    plt.title(kernel)
    x_sv, y_sv = clf.support_vectors_[:, 0], clf.support_vectors_[:, 1]
    plt.scatter(x_sv, y_sv, s=80, facecolors='none', zorder=10)
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)

    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)

    # plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())

plt.show()
