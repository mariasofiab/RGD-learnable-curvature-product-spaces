'''Tests'''
import RSGDlibrary
from RSGDlibrary import RSGD, everageDistortion
from costruisciGrafo import generaGrafo
import manifolds
from manifolds import plotOnProduct
from matplotlib import pyplot as plt

manifolds.plotOption = 3

def testArticolo(dimS, dimD, dimE):
    lr = 1e0
    epochs = 100
    nAlberi = 3
    G = generaGrafo(nAlberi) / 1
    # Costruisco P
    S = manifolds.SphericModel(dimS, 1)
    D = manifolds.PoincareModel(dimD, 1)
    E = manifolds.EuclideanModel(dimE)
    factors = []
    if dimS > 0: factors.append(S)
    if dimD > 0: factors.append(D)
    if dimE > 0: factors.append(E)
    P = manifolds.Product(factors)
    # Learning
    opt = RSGD(P, nAlberi * 4, lr=lr, X = None)
    X, P, devg = RSGDlibrary.learning(opt, G, epochs, no_curv = False,
                                      momento = True)
    return X, P, devg
G = generaGrafo(3, plot=True)
X, P, devg = testArticolo(0, 6, 0)

'''
Dimensioni  NaN con 3 alberi    NaN con 9 alberi
20, 0, 20                               
1, 1, 1      
10, 1, 10           
0, 2, 0             
1, 2, 0             x
2, 2, 0             x                   x
2, 2, 1             x                   
2, 2, 2             x                   x
'''