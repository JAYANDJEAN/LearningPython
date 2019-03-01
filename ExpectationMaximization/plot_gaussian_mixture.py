# coding:utf-8


from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture

mu_list = [.5, 1.0, 1.5]
N = 10
data_lists = [np.random.normal(loc=mu, scale=0.1, size=int(mu * N)) for mu in mu_list]
data_list = [j for temp in data_lists for j in temp]
data_temp = np.array(data_list)
data = data_temp.reshape(data_temp.shape[0], 1)
zero = 0.0 * np.random.normal(loc=0, scale=0.1, size=int(3 * N))


def gauss(x, mu, s):
    left = 1 / (np.sqrt(2 * np.pi * s))
    right = np.exp(-np.power((x - mu), 2) / 2 / s)
    return left * right


def draw(num):
    x = np.arange(0, 2, 0.001)

    # funcs = map(lambda j: lambda i: gm.weights_[j] * gauss(i, gm.means_[j][0], gm.covariances_[j][0][0]), range(num))
    funcs = [lambda i, j=n: gm.weights_[j] * gauss(i, gm.means_[j][0], gm.covariances_[j][0][0]) for n in range(num)]
    mix = np.array([list(map(f, x)) for f in funcs]).sum(axis=0)

    plt.figure(1, figsize=(10, 5))

    plt.plot(x, mix, 'b', linewidth=2)

    for j in range(num):
        y = map(lambda i: gm.weights_[j] * gauss(i, gm.means_[j][0], gm.covariances_[j][0][0]), x)
        plt.plot(x, y, 'r--', linewidth=1)

        num_point = 300
        alpha = 5
        x0 = [gm.means_[j][0] - alpha * gm.covariances_[j][0][0]] * num_point
        x1 = [gm.means_[j][0] + alpha * gm.covariances_[j][0][0]] * num_point
        y0 = np.arange(0, 3, 0.01)
        plt.plot(x0, y0, 'r--', linewidth=1)
        plt.plot(x1, y0, 'r--', linewidth=1)

    plt.scatter(data_list, zero, edgecolors='k', cmap=plt.cm.Paired)

    plt.show()


num = 3
gm = mixture.GaussianMixture(n_components=num)
gm.fit(data)
print gm.weights_, gm.means_, gm.covariances_

draw(num)
