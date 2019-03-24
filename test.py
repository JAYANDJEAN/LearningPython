# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg
from scipy.sparse import csc_matrix, csgraph
from sklearn.cluster import SpectralClustering, DBSCAN
import matplotlib.pyplot as plt

edges = [('1', 'a', 0.5),
         ('1', 'd', 0.6),
         ('2', 'a', 0.7),
         ('2', 'd', 0.7),
         ('2', 'b', 0.3),
         ('3', 'b', 0.3),
         ('3', 'c', 0.7),
         ('3', 'e', 0.7),
         ('4', 'c', 0.5),
         ('4', 'e', 0.6)]
nodes = []


def standardization(_d):
    mu = np.mean(_d, axis=0)
    sigma = np.std(_d, axis=0)
    return (_d - mu) / sigma


for e in edges:
    nodes.append(e[0])
    nodes.append(e[1])
nodes = list(set(nodes))

n_edge = len(edges)
n_node = len(nodes)
nodes_map = dict(zip(nodes, range(n_node)))

_row = np.zeros(shape=(n_edge,))
_col = np.zeros(shape=(n_edge,))
_data = np.zeros(shape=(n_edge,))

for idx, edge in enumerate(edges):
    _row[idx] = nodes_map[edge[0]]
    _col[idx] = nodes_map[edge[1]]
    _data[idx] = edge[2]

row = np.concatenate((_row, _col), axis=0)
col = np.concatenate((_col, _row), axis=0)
data = np.concatenate((_data, _data), axis=0)

mart = csc_matrix((data, (row, col)), shape=(n_node, n_node))

lap = csgraph.laplacian(mart, normed=False)
eigenvalues, eigenvectors = linalg.eig(lap.todense())
k = int(n_node / 3) if n_node > 6 else 2
sorted_indices = np.argsort(eigenvalues)

topk_evecs = eigenvectors[:, sorted_indices[:k]]
topk_evecs = standardization(topk_evecs)
clustering = DBSCAN(eps=0.7, min_samples=2)
c_res = clustering.fit_predict(topk_evecs)

print sorted(zip(nodes, c_res), key=lambda x: x[1])
