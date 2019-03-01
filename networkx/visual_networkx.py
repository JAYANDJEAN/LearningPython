# coding:utf-8
import re
import sys

import matplotlib.pyplot as plt
from networkx import nx

G = nx.DiGraph()

with open('wiki-Vote.txt') as f:
    data = f.readlines()
    data = [i.split('\r')[0] for i in data]
    data = [(i.split('\t')[0], i.split('\t')[1]) for i in data]

num_to_show = 1000

node_index = {}
for node_from, node_to in data[0:num_to_show]:
    if node_from not in node_index:
        node_index[node_from] = 0
        G.add_node(node_from)
    if node_to not in node_index:
        node_index[node_to] = 0
        G.add_node(node_to)
    G.add_edge(node_from, node_to)

UG = G.to_undirected()
options = {
    'node_color': 'black',
    'node_size': 1,
    'line_color': 'grey',
    'linewidths': 0,
    'width': 0.1,
}

nx.draw_circular(UG, **options)
plt.show()
