import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import datasets
from sklearn.neural_network import BernoulliRBM


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


digits = datasets.load_digits()
X = np.asarray(digits.data, 'float32')
X, Y = nudge_dataset(X, digits.target)
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

n_sample, n_feature = X.shape
n_split = int(n_sample * 0.8)
order = np.random.permutation(n_sample)

X_train = X[order][0:n_split, :]
Y_train = Y[order][0:n_split]
X_test = X[order][n_split:, :]
Y_test = Y[order][n_split:]

rbm = BernoulliRBM(random_state=0, verbose=True)
rbm.learning_rate = 0.06
rbm.n_iter = 20
rbm.n_components = 100
rbm.fit(X_train, Y_train)
X_new = rbm.fit_transform(X)
print X_new.shape

plt.figure(figsize=(8, 8))
for i in range(100):
    plt.subplot(10, 10, i + 1)
    comp = X_new[i:i + 1, :]
    plt.imshow(comp.reshape((10, 10)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()
