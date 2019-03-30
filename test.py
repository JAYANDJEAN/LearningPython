# coding:utf-8
from sklearn.cluster import SpectralClustering, DBSCAN
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.linalg.graphmatrix import adjacency_matrix
from networkx.linalg.laplacianmatrix import laplacian_matrix
from numpy import linalg

data = [
    ('1', 'a', {'w': 0.5}),
    ('1', 'd', {'w': 0.6}),
    ('2', 'a', {'w': 0.7}),
    ('2', 'd', {'w': 0.7}),
    ('2', 'b', {'w': 0.3}),
    ('3', 'b', {'w': 0.3}),
    ('3', 'c', {'w': 0.7}),
    ('3', 'e', {'w': 0.7}),
    ('4', 'c', {'w': 0.5}),
    ('4', 'e', {'w': 0.6})]

pos = {'1': [0.2, 0.4], '2': [0.4, 0.4],
       '3': [0.6, 0.4], '4': [0.8, 0.4],
       'b': [0.5, 0.2],
       'a': [0.1, 0.2], 'd': [0.3, 0.2],
       'c': [0.7, 0.2], 'e': [0.9, 0.2]}


class SpectralCluster(object):
    def __init__(self, data, pos):
        self.G = nx.Graph()
        self.G.add_edges_from(data)
        self.pos = pos

        self.fig = plt.figure(figsize=(15, 6))
        plt.subplots_adjust(left=0.05, bottom=0.10, right=0.95, top=0.90, wspace=0, hspace=0)

    def standardization(self, _data):
        mu = np.mean(_data, axis=1)
        sigma = np.std(_data, axis=1)
        return (_data - mu) / sigma

    def draw(self):
        colors = ['c', 'm', 'g', 'y', 'r', 'k']

        plt.subplot(131)
        nx.draw_networkx_nodes(self.G, self.pos, node_size=500)
        edges = [(u, v) for (u, v, d) in self.G.edges(data=True)]
        edges_width = [10 * d['w'] for (u, v, d) in self.G.edges(data=True)]
        nx.draw_networkx_edges(self.G, self.pos, edgelist=edges, width=edges_width)
        nx.draw_networkx_labels(self.G, self.pos, font_size=20, font_family='sans-serif')
        plt.axis('off')

        plt.subplot(132)
        lap_matrix = laplacian_matrix(self.G, weight='w').todense()
        node_list = self.G.nodes()
        k = 3
        eigenvalues, eigenvectors = linalg.eig(lap_matrix)
        sorted_indices = np.argsort(eigenvalues)
        topk_evecs = eigenvectors[:, sorted_indices[:k]]
        topk_evecs = self.standardization(topk_evecs)
        x = [np.array(i)[0][0] for i in topk_evecs]
        y = [np.array(i)[0][1] for i in topk_evecs]
        plt.scatter(x, y, c='r', s=50)
        for i, txt in enumerate(node_list):
            plt.annotate(txt, (x[i], y[i]))

        plt.subplot(133)
        clustering = DBSCAN(eps=0.7, min_samples=2)
        c_res = clustering.fit_predict(topk_evecs)
        dict_res = {}
        for c, n in zip(c_res, node_list):
            if c in dict_res:
                dict_res[c].append(n)
            else:
                dict_res[c] = [n]
        for idx, i in enumerate(dict_res):
            nx.draw_networkx_nodes(self.G, self.pos
                                   , nodelist=dict_res[i]
                                   , node_size=500
                                   , node_color=colors[idx])

        nx.draw_networkx_edges(self.G, self.pos, edgelist=edges, width=edges_width)
        nx.draw_networkx_labels(self.G, self.pos, font_size=20, font_family='sans-serif')
        plt.axis('off')

        plt.show()


sc = SpectralCluster(data, pos)
sc.draw()
