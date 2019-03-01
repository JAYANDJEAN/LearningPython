from __future__ import print_function
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
from matplotlib.finance import quotes_historical_yahoo_ochl
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from hmmlearn.hmm import GaussianHMM


date1 = datetime.date(2014, 10, 1)
date2 = datetime.date(2016, 10, 1)
quotes = quotes_historical_yahoo_ochl("BABA", date1, date2)

dates = np.array([q[0] for q in quotes], dtype=int)
close_v = np.array([q[2] for q in quotes])
volume = np.array([q[5] for q in quotes])[1:]

diff = close_v[1:] - close_v[:-1]
dates = dates[1:]
close_v = close_v[1:]

X = np.column_stack([diff, volume])

n_components = 5
model = GaussianHMM(n_components,
                    covariance_type="diag",
                    n_iter=1000)

print(X.shape)

model.fit(X)

hidden_states = model.predict(X)

print("Transition matrix")
print(model.transmat_)


for i in range(n_components):
    print("%dth hidden state" % i)
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))

years = YearLocator()   # every year
months = MonthLocator()  # every month
yearsFmt = DateFormatter('%Y')

fig = plt.figure()
ax = fig.add_subplot(111)
#ax2 = fig.add_subplot(212)



for i in range(n_components):
    # use fancy indexing to plot data in each stateo
    idx = (hidden_states == i)
    ax.plot_date(dates[idx], close_v[idx], '.')
    ax.plot_date(dates[idx], close_v[idx], '-')

ax.legend()

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
ax.autoscale_view()
# format the coords message box
ax.fmt_xdata = DateFormatter('%Y-%m-%d')

ax.fmt_ydata = lambda x: '$%1.2f' % x
ax.grid(True)
plt.ylim(50,140)
ax.set_xlabel('Year')
ax.set_ylabel('Closing Volume')
fig.autofmt_xdate()
plt.show()
