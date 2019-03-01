import cvxopt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# we create 40 separable points
# np.random.seed(0)
X = np.vstack(([np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]))
y = [0] * 20 + [1] * 20

plt.figure(1, figsize=(12, 5))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)


class SupportVectorMachine(object):
    def __init__(self, X, y, c=0.1):
        self._X = X
        self.unique_y = np.unique(y)
        assert len(self.unique_y) == 2
        self._y = [-1.0 if i == self.unique_y[0] else 1.0 for i in y]
        self._y = np.array(self._y).reshape(len(self._y), 1)
        self._c = c
        self._MIN_MULTIPLIER = 0.01
        self._bias = 0.0
        self._support_multipliers = [0]
        self._support_vectors = [0]
        self._support_vector_labels = [0]

    def train(self):
        X = self._X
        y = self._y
        lagrange_multipliers = self._compute_multipliers(X, y)
        support_vector_indices = lagrange_multipliers > self._MIN_MULTIPLIER
        self._support_multipliers = lagrange_multipliers[support_vector_indices]
        self._support_vectors = X[support_vector_indices]
        self._support_vector_labels = y[support_vector_indices]

        self._bias = np.mean(
                [y_k - self._compute_bias(x_k)
                 for (y_k, x_k) in
                 zip(self._support_vector_labels, self._support_vectors)])

        return lagrange_multipliers



    def predict(self, x):

        result = self._bias
        for z_i, x_i, y_i in zip(self._support_multipliers, self._support_vectors, self._support_vector_labels):
            result += z_i * y_i * self._kernel(x_i, x)

        return result

    def _compute_bias(self, x):
        bias=0.0
        for z_i, x_i, y_i in zip(self._support_multipliers, self._support_vectors, self._support_vector_labels):
            bias += z_i * y_i * self._kernel(x_i, x)

        return bias

    def _kernel(self, x1, x2):
        return np.inner(x1, x2)

    def _gram_matrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)
        return K

    def _compute_multipliers(self, X, y):
        n_samples, n_features = X.shape

        K = self._gram_matrix(X)
        # Solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \coneleq h
        #  Ax = b

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        # -a_i \leq 0
        # TODO(tulloch) - modify G, h so that we have a soft-margin classifier
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        # a_i \leq c
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        return np.ravel(solution['x'])


###################################################

mysvm = SupportVectorMachine(X, y)
lagrange_multipliers = mysvm.train()
print lagrange_multipliers
print mysvm._bias
sv = mysvm._support_vectors

plt.sca(ax1)
plt.scatter(sv[:, 0], sv[:, 1], s=80, facecolors='none', zorder=10)
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

####################################################

clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, y)

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy + a * margin
yy_up = yy - a * margin

plt.sca(ax2)
plt.title('unreg')
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none', zorder=10)
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

plt.axis('tight')

plt.xlim(-6, 6)
plt.ylim(-6, 6)

plt.show()
