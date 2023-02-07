import networkx as nx
from torch import cos, sin, pi
from torch import tensor as tn
from matplotlib import pyplot as plt

def generaGrafo(n, plot = False):
    G = nx.Graph()
    G.add_nodes_from(range(4*n))
    for i in range(n):
        G.add_edge(i, (i+1)%n)
        G.add_edge(i, i+n)
        G.add_edge(i+n, i + 2*n)
        G.add_edge(i+n, i + 3*n)
    if plot:
        pos = dict()
        for i in range(n):
            alpha = tn(2 * pi / (n*4))
            theta = tn(2 * pi * i / n)
            pos[i] = (cos(theta), sin(theta))
            pos[i+n] = (2*cos(theta), 2*sin(theta))
            pos[i+2*n] = (3*cos(theta-alpha), 3*sin(theta-alpha))
            pos[i+3*n] = (3*cos(theta+alpha), 3*sin(theta+alpha))
        nx.draw(G, pos, with_labels=True)
        plt.savefig('resultGrafo', ppi = 600)
        plt.show()
    Gdict = dict(nx.all_pairs_shortest_path_length(G))
    M = tn([[float(Gdict[i][j]) for i in range(G.size())] for j in range(G.size())])
    return M