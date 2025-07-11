"""
Interactively highlight nodes and edges
---------------------------------------

Run this with an interactive matplotlib backend!

Clicking on a node will hi-light it and it's edges
"""
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from netgraph import Graph, InteractiveGraph, EditableGraph
from experiment_for_anm_rule import *

seed=3444
randomState = np.random.RandomState(seed)
in_degree=1
num_self_node=2
rom=1/3
dim=15
B_true = randomGraph(dim, in_degree, randomState)
DAG = randomMGraph(B_true,rom=rom, num_self_node=num_self_node, randomState=randomState)
G = DAG2OracleANM(DAG)



n = len(G) // 2
labels = []
for i in range(n):
    labels.append("x" + str(i))
for i in range(n):
    labels.append("R" + str(i))
df = pd.DataFrame(G, index=labels, columns=labels)
df = df.loc[(df.sum(0) != 0) | (df.sum(1) != 0), (df.sum(0) != 0) | (df.sum(1) != 0)]

graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)

plot_instance = InteractiveGraph(graph,arrows=True,node_labels=True,edge_width=2, node_label_fontdict=dict(size=10),node_size=5)
plt.show()
# pos = graphviz_layo-
# ut(G2, prog="dot")

##pos = nx.nx_pydot.graphviz_layout(graph)
# pos = nx.spring_layout(G2,iterations=20)

# nx.draw_circular(G2, with_labels=True)
#art=nx.draw_networkx(graph, pos, with_labels=True)

#graph = nx.barbell_graph(10, 14)
