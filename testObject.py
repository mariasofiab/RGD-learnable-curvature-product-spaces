'''Tests'''
import RSGDlibrary
from RSGDlibrary import RSGD, everageDistortion
from costruisciGrafo import generaGrafo
import manifolds
from manifolds import plotOnProduct
from matplotlib import pyplot as plt

def testArticolo(dimS, dimD, dimE):
    lr = 1e0
    epochs = 100
    nAlberi = 9
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
    opt = RSGD(P, nAlberi * 4, lr=lr)
    X, K, losses, curvs = RSGDlibrary.learning(opt, G, epochs, no_curv = False,
                                               momento = True)
    return X, K, losses, P, curvs
G = generaGrafo(9, plot=True) / 1
plotOption = 0
X0, K0, losses0, P0, curvs0 = testArticolo(3,0,0)
X1, K1, losses1, P1, curvs1 = testArticolo(0,3,0)
X2, K2, losses2, P2, curvs2 = testArticolo(0,0,3)
X3, K3, losses3, P3, curvs3 = testArticolo(1,2,0)
X4, K4, losses4, P4, curvs4 = testArticolo(0,2,1)
X5, K5, losses5, P5, curvs5 = testArticolo(2,0,1)
X6, K6, losses6, P6, curvs6 = testArticolo(1,1,1)

# Plot loss
plt.clf()
plt.plot(losses0[-300:], label = '$'+str(P0)+'$ - $D_{avg}$: %.2f'%everageDistortion(X0, P0, G))
plt.plot(losses1[-300:], label = '$'+str(P1)+'$ - $D_{avg}$: %.2f'%everageDistortion(X1, P1, G))
plt.plot(losses2[-300:], label = '$'+str(P2)+'$ - $D_{avg}$: %.2f'%everageDistortion(X2, P2, G))
plt.plot(losses3[-300:], label = '$'+str(P3)+'$ - $D_{avg}$: %.2f'%everageDistortion(X3, P3, G))
plt.plot(losses4[-300:], label = '$'+str(P4)+'$ - $D_{avg}$: %.2f'%everageDistortion(X4, P4, G))
plt.plot(losses5[-300:], label = '$'+str(P5)+'$ - $D_{avg}$: %.2f'%everageDistortion(X5, P5, G))
plt.plot(losses6[-300:], label = '$'+str(P6)+'$ - $D_{avg}$: %.2f'%everageDistortion(X6, P6, G))
plt.legend(); plt.show()

# Plot Embedding
# plotOption = 2
# plotOnProduct(X0, P0, G)
# plotOnProduct(X1, P1, G)
# plotOnProduct(X2, P2, G)
# plotOnProduct(X3, P3, G)
# plotOnProduct(X4, P4, G)
# plotOnProduct(X5, P5, G)
# plotOnProduct(X6, P6, G)

# Plot Curvs
# plt.clf()
# for fact in range(len(curvs0[0])):
#     plt.plot([curvs0[i][fact] for i in range(len(curvs0))], label = '$'+str(P0.factors[fact])+'$')
# plt.legend(); plt.show()
# for fact in range(len(curvs1[0])):
#     plt.plot([curvs1[i][fact] for i in range(len(curvs1))], label = '$'+str(P1.factors[fact])+'$')
# plt.legend(); plt.show()
# for fact in range(len(curvs2[0])):
#     plt.plot([curvs2[i][fact] for i in range(len(curvs2))], label = '$'+str(P2.factors[fact])+'$')
# plt.legend(); plt.show()
# for fact in range(len(curvs3[0])):
#     plt.plot([curvs3[i][fact] for i in range(len(curvs3))], label = '$'+str(P3.factors[fact])+'$')
# plt.legend(); plt.show()
# for fact in range(len(curvs4[0])):
#     plt.plot([curvs4[i][fact] for i in range(len(curvs4))], label = '$'+str(P4.factors[fact])+'$')
# plt.legend(); plt.show()
# for fact in range(len(curvs5[0])):
#     plt.plot([curvs5[i][fact] for i in range(len(curvs5))], label = '$'+str(P5.factors[fact])+'$')
# plt.legend(); plt.show()
# for fact in range(len(curvs6[0])):
#     plt.plot([curvs6[i][fact] for i in range(len(curvs6))], label = '$'+str(P6.factors[fact])+'$')
# plt.legend(); plt.show