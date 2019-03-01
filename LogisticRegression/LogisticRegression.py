# coding:utf-8
from __future__ import division
import numpy as np


class LogRegression(object):
    def __init__(self, X, y, c=0.00001, regularization='l2', alpha=1, lamda=1):

        self.c = c
        self.regularization = regularization
        self.alpha = alpha
        self.lamda = lamda
        self.X = X
        self.unique_y_names = np.unique(y)
        y_0_1 = [0 if i == self.unique_y_names[0] else 1 for i in y]
        self.y = y_0_1
        self.n_sample, self.n_feature = X.shape
        self.theta_final = np.zeros(self.n_feature)

    def train(self):
        n_sample, n_feature = self.n_sample, self.n_feature
        c = self.c
        theta_init = np.zeros(n_feature)
        theta_new = theta_init
        cost0 = 0;
        cost1 = 1
        k = 0
        while np.abs(cost1 - cost0) > c:
            cost0 = self._computeCost(theta_new)
            theta_new = self._computeTheta(theta_new)
            cost1 = self._computeCost(theta_new)
            print cost1
            k += 1
        print "number_of_iteration:%i" % k
        self.theta_final = theta_new
        return 0

    def predict(self, x):

        x = np.array(x, dtype=np.double)
        s = self._sigmoid(np.dot(x, self.theta_final))

        result = [self.unique_y_names[0] if i < 0.5 else self.unique_y_names[1] for i in s]
        return result

    def _sigmoid(self, x):

        x = np.array(x, dtype=np.double)
        return 1 / (1 + np.exp(-x))

    def _computeCost(self, theta):

        lamda = self.lamda;
        cost = 0.0
        n_sample, n_feature = self.n_sample, self.n_feature
        if self.regularization == 'l1':
            reg_parameter = np.dot(lamda / (2 * n_sample), np.sum(np.abs(theta)))
        if self.regularization == 'l2':
            reg_parameter = np.dot(lamda / (2 * n_sample), np.sum(np.dot(theta, theta)))
        if self.regularization == '':
            reg_parameter = 0.0

        for i in range(n_sample):
            xi = self.X[i]
            yi = self.y[i]
            cost += 1 / n_sample * np.log(1 + np.exp(-yi * np.dot(xi, theta)))
        cost = cost + reg_parameter
        return cost

    def _computeTheta(self, theta):

        n_sample, n_feature = self.n_sample, self.n_feature
        alpha = self.alpha
        lamda = self.lamda if (len(self.regularization) > 0) else 0.0
        grad = 0.0

        for i in range(n_sample):
            xi = self.X[i]
            yi = self.y[i]
            grad += 1 / n_sample * (yi - self._sigmoid(np.dot(theta, xi))) * xi

        grad += lamda / n_sample * theta
        theta_new = theta + alpha * grad

        return theta_new


