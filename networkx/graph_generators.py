import matplotlib.pyplot as plt
import networkx as nx

G = nx.planted_partition_graph(3, 20, 0.9, 0.01)

options = {
    'node_color': 'black',
    'node_size': 30,
    'line_color': 'grey',
    'linewidths': 0,
    'width': 0.1,
}
nx.draw(G, **options)
plt.show()
