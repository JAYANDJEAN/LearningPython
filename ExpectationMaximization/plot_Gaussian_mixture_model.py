# coding:utf-8

from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GMM(object):
    def __init__(self, means_list, covariances_list, n_components, xlim, ylim):
        self.means_list = means_list
        self.covariances_list = covariances_list
        self.n_components = n_components
        self.xlim = xlim
        self.ylim = ylim

    def generateData(self, means, covariances, size):
        return multivariate_normal(means, covariances).rvs(size=size)

    def fit(self):
        data_list = [self.generateData(self.means_list[i], self.covariances_list[i], 10) for i in
                     range(self.n_components)]
        data = np.vstack(tuple(data_list))
        gmm = GaussianMixture(n_components=self.n_components)
        gmm.fit(data)

        x, y = np.mgrid[self.xlim:self.ylim:.01, self.xlim:self.ylim:.01]
        pos = np.dstack((x, y))
        data_mix = sum([gmm.weights_[i] * multivariate_normal(gmm.means_[i], gmm.covariances_[i]).pdf(pos) for i in
                        range(self.n_components)])
        return data_list, data_mix

    def draw(self, data_list, data_mix):
        plt.figure(1, figsize=(11, 5))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        for data in data_list:
            plt.sca(ax1)
            plt.scatter(data[:, 0], data[:, 1], edgecolors='k')
        plt.xlim(self.xlim, self.ylim)
        plt.ylim(self.xlim, self.ylim)

        x, y = np.mgrid[self.xlim:self.ylim:.01, self.xlim:self.ylim:.01]
        plt.sca(ax2)
        plt.contourf(x, y, data_mix)
        plt.xlim(self.xlim, self.ylim)
        plt.ylim(self.xlim, self.ylim)

        plt.show()


means_list = [[0.5, 0.5], [1.5, 1.6]]
covariances_1 = np.array([[.1, .1],
                          [.1, .2]])
covariances_2 = np.array([[.7, .6],
                          [.6, .9]])
covariances_list = [covariances_1, covariances_2]

test = GMM(means_list, covariances_list, 2, -2, 5)
data_list, data_mix = test.fit()
test.draw(data_list, data_mix)
