import networkx as nx
from torch import cos, sin, pi
from torch import tensor as tn
from matplotlib import pyplot as plt
import math

def graph(key, plot = False, n = None):
    G = nx.Graph(); pos= dict();
    if key == 'ciclo':
        N = 36
        if not n is None: N = n
        for i in range(N):
            G.add_edge(i, (i+1)%N)
        if plot:
            for i in range(N):
                alpha = tn(2 * pi / (N*4))
                theta = tn(2 * pi * i / N)
                pos[i] = (cos(theta), sin(theta))
    elif key == 'anello':
        N = 9
        if not n is None: N = n//4
        G.add_nodes_from(range(4*N))
        for i in range(N):
            G.add_edge(i, (i+1)%N)
            G.add_edge(i, i+N)
            G.add_edge(i+N, i + 2*N)
            G.add_edge(i+N, i + 3*N)
        if plot:
            for i in range(N):
                alpha = tn(2 * pi / (N*4))
                theta = tn(2 * pi * i / N)
                pos[i] = (cos(theta), sin(theta))
                pos[i+N] = (2*cos(theta), 2*sin(theta))
                pos[i+2*N] = (3*cos(theta-alpha), 3*sin(theta-alpha))
                pos[i+3*N] = (3*cos(theta+alpha), 3*sin(theta+alpha))
    elif key == 'albero':
        h = 5; b = 2;
        if not n is None: h = int(math.log(n,b))
        N = int((b**(h+1)-1) / (b-1))
        if plot: pos[0]=(tn(.5),tn(0))
        for alt in range(h-1):# I for sono sbagliati, funzionano bene solo per b = 2. Sistemali
            for dad in range(b**alt):
                for son in range(b):
                    sxDad = (b**(alt)-1) / (b-1)
                    sxSon = (b**(alt+1)-1) / (b-1)
                    i = int(sxDad + dad)
                    j = int(sxSon + dad*b + son)
                    G.add_edge(i, j)
                    if plot:
                        dx = 1 / (b**(alt+1)+1)
                        pos[j] = (tn(dx * (1 + dad * b + son)), -tn(alt+1))
    elif key == 'linea':
        N = 36
        if not n is None: N = n
        G.add_nodes_from(range(N))
        for i in range(N-1):
            G.add_edge(i, i+1)
        if plot:
            for i in range(N):
                pos[i] = (tn(i),tn(0.))
    else:
        raise ValueError('Invalid key')
    if plot:
        nx.draw(G, pos, with_labels=True)
        plt.savefig('Immagini/'+key, dpi = 600)
        plt.show()
    Gdict = dict(nx.all_pairs_shortest_path_length(G))
    M = tn([[float(Gdict[i][j]) for i in range(len(G.nodes))] for j in range(len(G.nodes))])
    return M
        
# G1 = graph('ciclo',True)
# G2 = graph('anello',True)
#G3 = graph('albero',True)
# G3 = graph('linea',True)

