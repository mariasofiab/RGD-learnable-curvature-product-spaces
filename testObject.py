'''Tests'''
import RSGDlibrary
from RSGDlibrary import RSGD
from costruisciGrafo import generaGrafo
import manifolds
from matplotlib import pyplot as plt

def testArticolo(lr = 1e-2):
    nPunti = 9; epochs = 50
    G = generaGrafo(nPunti)
    plt.clf()
    # Costruisco P
    S1 = manifolds.SphericModel(1, 1)
    D2 = manifolds.PoincareModel(2, 1)
    P = manifolds.Product([S1,D2])
    # Learning
    opt = RSGD(P, nPunti * 4, lr=lr)
    X, K, losses = RSGDlibrary.learning(opt, G, epochs, no_curv = False)
    plt.plot(losses)
    plt.show()
    return X, K, losses

X, K, losses = testArticolo()