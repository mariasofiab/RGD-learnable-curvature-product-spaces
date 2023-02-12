'''Tests'''
import RSGDlibrary
from RSGDlibrary import RSGD, everageDistortion
from costruisciGrafo import generaGrafo
import manifolds
from manifolds import plotOnProduct
from matplotlib import pyplot as plt
from torch import tensor as tn

manifolds.plotOption = 3

def testArticolo(dimS, dimD, dimE, nAlberi=9):
    lr = 1e-1
    epochs = 1000
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
res3 = []; res9 = [];
dimss = [[1,1,1],[10,1,10],[0,2,0],[1,2,0],[2,2,0],[2,2,1],[2,2,2],[10,10,10]]
for test, dims in enumerate(dimss):
    print('Test %d / %d'%(test, len(dimss)))
    X, P, devg = testArticolo(dims[0], dims[1], dims[2], nAlberi = 3)
    res3.append({'Dims':dims,
                 'Nan':bool(tn(P.getCurvatures()).isnan().any()),
                 'Devg':float(devg)})
    X, P, devg = testArticolo(dims[0], dims[1], dims[2], nAlberi = 9)
    res9.append({'Dims':dims,
                 'Nan':bool(tn(P.getCurvatures()).isnan().any()),
                 'Devg':float(devg)})
print(res3)
print(res9)