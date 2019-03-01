import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge

def f(x): return x * np.sin(x)

x_plot = np.linspace(0, 10, 100).reshape(100, 1)

x = np.linspace(0, 10, 100)
order = np.random.permutation(100)
x = x[order[:20]].reshape(20, 1)
y = f(x)

plt.plot(x_plot, f(x_plot), label="ground truth")
plt.scatter(x, y, label="training points")

for degree in [3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(x, y)
    y_plot = model.predict(x_plot)
    #plt.plot(x_plot, y_plot, label="degree %d" % degree)

kr = KernelRidge(kernel='poly', degree=4)
kr.fit(x, y)
y_plot = kr.predict(x_plot)
plt.plot(x_plot, y_plot, label="KernelRidge")

plt.legend(loc='lower left')

plt.show()
