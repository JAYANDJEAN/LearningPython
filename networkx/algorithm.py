import matplotlib.pyplot as plt
import networkx as nx

G0 = nx.planted_partition_graph(3, 20, 0.9, 0.01)
G1 = nx.path_graph(3)

# Compute degree assortativity of graph.
r0 = nx.degree_assortativity_coefficient(G0)
r1 = nx.degree_assortativity_coefficient(G1)

print nx.degree_histogram(G0)

print nx.average_clustering(G0)