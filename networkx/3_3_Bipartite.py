import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from networkx.algorithms import centrality

B = nx.Graph()

B.add_nodes_from([1, 2, 3, 4], bipartite=0)
B.add_nodes_from(['a', 'b', 'c', 'd'], bipartite=1)

edges_list = [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'), (2, 'c'), (3, 'c'), (3, 'd'), (4, 'c'), (4, 'd')]
B.add_edges_from(edges_list)

pos = nx.spring_layout(B)
sets = bipartite.sets(B)
colors = ['r', 'b']

c = bipartite.robins_alexander_clustering(B)

print centrality.degree_centrality(B)

for i in range(len(sets)):
    nx.draw_networkx_nodes(B, pos, nodelist=list(sets[i]), node_color=colors[i])

nx.draw_networkx_edges(B, pos)
plt.show()
