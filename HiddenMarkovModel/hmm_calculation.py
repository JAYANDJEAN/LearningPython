# coding:utf-8
'''
Created on 2016年11月4日

@author: Jay
'''
from hmmlearn.hmm import MultinomialHMM
import numpy as np

# n_components 代表隐状态的个数
# n_symbols 代表可能的观察数
model_multinomial = MultinomialHMM(n_components=4)

# 概率转移矩阵：n_components*n_components
transition_matrix = np.array([[0.2, 0.6, 0.15, 0.05],
                              [0.2, 0.3, 0.3, 0.2],
                              [0.05, 0.05, 0.7, 0.2],
                              [0.005, 0.045, 0.15, 0.8]])
model_multinomial.transmat_ = transition_matrix

# 初始状态概率：n_components*1
initial_state_prob = np.array([0.1, 0.4, 0.4, 0.1])
model_multinomial.startprob_ = initial_state_prob

# 观测概率矩阵：n_components*n_symbols
emission_prob = np.array([[0.045, 0.15, 0.2, 0.6, 0.005],
                          [0.2, 0.2, 0.2, 0.3, 0.1],
                          [0.3, 0.1, 0.1, 0.05, 0.45],
                          [0.1, 0.1, 0.2, 0.05, 0.55]])
model_multinomial.emissionprob_ = emission_prob

# model.sample返回观测数据和隐藏的状态数据：V是观测数据；H是隐藏数据
V, H = model_multinomial.sample(100)
