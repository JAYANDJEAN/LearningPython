import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# we create 40 separable points
# np.random.seed(0)
X = np.vstack(([np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]))
Y = [0] * 20 + [1] * 20

plt.figure(1, figsize=(12, 5))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)


for name, penalty, ax in (('unreg', 1, ax1), ('reg', 0.05, ax2)):
    clf = svm.SVC(kernel='linear', C=penalty)
    clf.fit(X, Y)

    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy + a * margin
    yy_up = yy - a * margin

    plt.sca(ax)
    plt.title(name)
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none', zorder=10)
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)

    plt.axis('tight')

    plt.xlim(-6, 6)
    plt.ylim(-6, 6)


plt.show()
