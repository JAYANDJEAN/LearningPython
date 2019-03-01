# coding:utf-8
import math
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import datetime

date_start = '20180601'
days_training = 3
hour_cur_day_list = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
hour_pred = 2
T = 24

# Plot results
fig, axes = plt.subplots(figsize=(16, 9), nrows=3, ncols=4)


# Generate sample data
def generate_data():
    file_name = 'data.txt'
    with open(file_name) as f:
        data_dict = {}
        data = f.readlines()
        data = [i.split('\n')[0] for i in data]
        data = [i.split(',') for i in data]
        data = sorted(data, key=lambda x: int(x[0]))
        for i in data:
            if i[1] in data_dict:
                data_dict[i[1]].append([i[2], i[3]])
            else:
                data_dict[i[1]] = [[i[2], i[3]]]

    datetime_start = datetime.datetime.strptime(date_start, '%Y%m%d')
    data_x = range((days_training + 1) * T)
    data_y = []

    for i in range(days_training + 1):
        date_now = datetime_start + datetime.timedelta(i)
        date_now_str = date_now.strftime('%Y%m%d')
        data = sorted(data_dict[date_now_str], key=lambda x: int(x[0]))
        data_y += [int(d[1]) for d in data]
    return data_x, data_y


# class
class PeriodicRegression(object):
    def __init__(self):
        self.fita = None
        self.fitb = None

    def func_base_period(self, x, c1, c2, c3, c4, c5, c6
                         , a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5):
        f1 = c1 \
             + c2 * x \
             + c3 * np.power(x, 2) \
             + c4 * np.power(x, 3) \
             + c5 * np.power(x, 0.5) \
             + c6 * np.exp(-x)
        f2 = a0 \
             + a1 * np.cos(2 * np.pi / T * x) + b1 * np.sin(2 * np.pi / T * x) \
             + a2 * np.cos(4 * np.pi / T * x) + b2 * np.sin(4 * np.pi / T * x) \
             + a3 * np.cos(6 * np.pi / T * x) + b3 * np.sin(6 * np.pi / T * x) \
             + a4 * np.cos(8 * np.pi / T * x) + b4 * np.sin(8 * np.pi / T * x) \
             + a5 * np.cos(10 * np.pi / T * x) + b5 * np.sin(10 * np.pi / T * x)
        return f1 * f2

    def fit(self, x, y):
        self.fita, self.fitb = optimize.curve_fit(self.func_base_period, x, y)

    def predict(self, x):
        if isinstance(x, list):
            return [
                self.func_base_period(i, self.fita[0], self.fita[1], self.fita[2], self.fita[3],
                                      self.fita[4], self.fita[5], self.fita[6], self.fita[7],
                                      self.fita[8], self.fita[9], self.fita[10], self.fita[11],
                                      self.fita[12], self.fita[13], self.fita[14], self.fita[15],
                                      self.fita[16])
                for i in x]
        else:
            return self.func_base_period(x, self.fita[0], self.fita[1], self.fita[2], self.fita[3],
                                         self.fita[4], self.fita[5], self.fita[6], self.fita[7],
                                         self.fita[8], self.fita[9], self.fita[10], self.fita[11],
                                         self.fita[12], self.fita[13], self.fita[14], self.fita[15],
                                         self.fita[16])


data_x, data_y = generate_data()
pr = PeriodicRegression()

for ind1 in range(3):
    for ind2 in range(4):
        hh = hour_cur_day_list[4 * ind1 + ind2]
        x_train = data_x[0:days_training * T + hh]
        y_train = data_y[0:days_training * T + hh]
        x_test = data_x[days_training * T + hh:days_training * T + hh + hour_pred]
        y_test = data_y[days_training * T + hh:days_training * T + hh + hour_pred]
        x_plot = np.linspace(0, (days_training + 1) * T, 1000)

        pr.fit(x_train, y_train)

        line = ''
        y_predict = pr.predict(x_test)
        for i in range(len(y_predict)):
            diff = float(y_predict[i] - y_test[i]) / y_test[i]
            line += str(hh + 1 + i) + " o'clock: " + 'diff is ' + str(format(diff, '.0%')) + '\n'

        axes[ind1, ind2].scatter(data_x, data_y, c='k', s=5, label='data for trainning')
        axes[ind1, ind2].scatter(x_test, y_test, c='r', s=20, label=line)
        axes[ind1, ind2].plot(x_plot, pr.predict(x_plot), color='turquoise', lw=2, label='Fourier')
        axes[ind1, ind2].legend(loc="best", scatterpoints=1, prop={'size': 8})

fig.tight_layout()
plt.text(2, 100, 'A tale of 2 subplots')
plt.savefig('te.png')
plt.show()
