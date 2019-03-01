from __future__ import division
import math
import matplotlib.pyplot as plt
import numpy as np


def gd(x, m, s):
    left = 1 / (math.sqrt(2 * math.pi) * s)
    right = math.exp(-math.pow(x - m, 2) / (2 * math.pow(s, 2)))
    return left * right


x = np.arange(0, 1, 0.001)

mu_c1_1 = 0.2
mu_c1_2 = 0.5
sigma_c1_1 = 0.07
sigma_c1_2 = 0.09

mu_c2 = 0.7
sigma_c2 = 0.05

p_c1 = 0.9
p_c2 = 0.1

p_x_c1 = [gd(i, mu_c1_1, sigma_c1_1) + gd(i, mu_c1_2, sigma_c1_2) for i in x]
p_x_c2 = [gd(i, mu_c2, sigma_c2) for i in x]
p_x = map(lambda i, j: i + j, p_x_c1, p_x_c2)

p_c1_x = map(lambda i, j: i * p_c1 / (i * p_c1 + j * p_c2), p_x_c1, p_x_c2)
p_c2_x = map(lambda i, j: j * p_c2 / (i * p_c1 + j * p_c2), p_x_c1, p_x_c2)

plt.figure(1, figsize=(12, 5))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

plt.sca(ax1)
plt.ylim(0.0, 12)
plt.plot(x, p_x_c1, 'b', linewidth=2)
plt.plot(x, p_x_c2, 'r', linewidth=2)
# plt.plot(x, p_x, 'g', linewidth=3)

text_p_x_c1 = '$p(x\mid C_1)$\n~N(' + str(mu_c1_1) + ',' + str(sigma_c1_1) + ')+N(' + str(mu_c1_2) + ',' + str(
    sigma_c1_2) + ')'
text_p_x_c2 = '$p(x\mid C_2)$\n~N(' + str(mu_c2) + ',' + str(sigma_c2) + ')'

plt.text(0.1, 6.5, text_p_x_c1, fontsize=15)
plt.text(0.6, 9, text_p_x_c2, fontsize=15)

###########################################

plt.sca(ax2)

plt.ylim(0.0, 1.2)
plt.plot(x, p_c1_x, 'b', linewidth=2)
plt.plot(x, p_c2_x, 'r', linewidth=2)

text_p_c1_x = '$p(C_1\mid x)$'
text_p_c2_x = '$p(C_2\mid x)$'
text_p_c1 = '$p(C_1)=$' + str(p_c1)
text_p_c2 = '$p(C_2)=$' + str(p_c2)

plt.text(0.2, 1.05, text_p_c1_x, fontsize=15)
plt.text(0.75, 1.05, text_p_c2_x, fontsize=15)
plt.text(0.1, 0.6, text_p_c1, fontsize=15)
plt.text(0.1, 0.5, text_p_c2, fontsize=15)

plt.show()
