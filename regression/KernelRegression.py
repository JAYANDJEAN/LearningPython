import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, ARDRegression
from sklearn import neighbors

rng = np.random.RandomState(0)

# Generate sample data
# X = 15 * rng.rand(100, 1)
# y = np.sin(X).ravel()
# y += 3 * (0.5 - rng.rand(X.shape[0]))  # add noise
data = pd.read_csv('data.txt', dtype='float32')
num = 105
y = data['cnt'].values[0:num]
y = y / max(y)
X = np.arange(len(y)).reshape(num, 1)

# Fit KernelRidge with parameter selection based on 5-fold cross validation

kr = GridSearchCV(KernelRidge()
                  , cv=5
                  , param_grid={"alpha": [1e0, 1e-1, 1e-2, 1e-3],
                                "kernel": [ExpSineSquared(l, p)
                                           for l in np.logspace(-2, 2, 10)
                                           for p in np.logspace(0, 2, 10)]})
kr.fit(X, y)
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1)
                   , cv=5
                   , param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                 "gamma": np.logspace(-2, 2, 5)})
svr.fit(X, y)
# mlp = GridSearchCV(MLPRegressor(hidden_layer_sizes=(100, 2), max_iter=1000)
#                    , cv=5
#                    , param_grid={"alpha": np.logspace(-5, 3, 5)})
# mlp.fit(X, y)
abr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                        n_estimators=1000, random_state=rng)
abr.fit(X, y)
gbr = GradientBoostingRegressor(loss='quantile', alpha=0.95,
                                n_estimators=250, max_depth=3,
                                learning_rate=.1, min_samples_leaf=9,
                                min_samples_split=9)
gbr.fit(X, y)
br = BayesianRidge(compute_score=True)
br.fit(X, y)
ardr = ARDRegression(compute_score=True)
ardr.fit(X, y)
knn = neighbors.KNeighborsRegressor(5, weights='distance')
knn.fit(X, y)

# Predict using kernel ridge
X_plot = np.linspace(0, num + 10, 10000)[:, None]
y_kr = kr.predict(X_plot)
y_svr = svr.predict(X_plot)
y_abr = abr.predict(X_plot)
y_gbr = gbr.predict(X_plot)
y_br = br.predict(X_plot)
y_ardr = ardr.predict(X_plot)
y_knn = knn.predict(X_plot)

# Plot results
fig = plt.figure(figsize=(10, 5))
lw = 2
plt.scatter(X, y, c='k', s=5, label='data')
plt.plot(X_plot, y_kr, color='turquoise', lw=lw, label='KRR (%s)' % kr.best_params_)
# plt.plot(X_plot, y_svr, color='r', lw=lw, label='SVR (%s)' % svr.best_params_)
plt.plot(X_plot, y_abr, color='g', lw=lw, label='AdaBoostRegressor')
# plt.plot(X_plot, y_gbr, color='k', lw=lw, label='GradientBoostingRegressor')
# plt.plot(X_plot, y_br, color='k', lw=lw, label='BayesianRidge')
# plt.plot(X_plot, y_ardr, color='k', lw=lw, label='ARDRegression')
# plt.plot(X_plot, y_knn, color='r', lw=lw, label='KNeighborsRegressor')
plt.xlabel('data')
plt.ylabel('target')

plt.title('Regression')
plt.legend(loc="best", scatterpoints=1, prop={'size': 8})
plt.show()
