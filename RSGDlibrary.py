import torch
from torch import tensor as tn
from torch import sqrt, cos, sin, cosh, sinh, arccos, tanh, atanh, arccosh, pi
import manifolds
from tqdm import tqdm

plot = True

'''
params = [
            {"params": X, "lr": 0.001},
            {"params": K, "lr": 0.01}
         ]
X is a list of tensors s.t. X[i] is the i° point and X[i][j] his j° coordinate.
K is a list of 1-tensors and K[i] is the magnitude of curvature of i° manifold.
'''

def grad(p: torch.Tensor, product: manifolds.Product):    
    M = product.inverseTensor(p)
    return torch.matmul(M, p.grad.data)

class RSGD(torch.optim.Optimizer):
    def __init__(self, product : manifolds.Product, numPoints, lr = 1e-2):
        X = product.randomPoints(n = numPoints, grad=True)
        K = [M.curvature for M in product.factors]
        params = [
                    {"params": X, "lr": lr},
                    {"params": K, "lr": lr}
                 ]
        super(RSGD, self).__init__(params, {})
        self.product= product

    def step(self):
        X = self.param_groups[0]["params"] # Points
        lr = self.param_groups[0]["lr"]
        for x in X:
            if x.grad is None:
                continue
            dx = grad(x, self.product) # Riemannian Gradient
            dx = self.product.projection(x, dx) # Projection
            dx.clamp_(-1., 1.)# Clipping
            dx.mul_(-lr) # Multiply for lr
            newx = self.product.expMap(x, dx) # Exponential Map
            x.data.copy_(newx) # Update
        K = self.param_groups[1]["params"] # Curvatures
        lr = self.param_groups[0]["lr"]
        for i in range(len(K)):
            k = K[i]
            if k.grad is None:
               continue
            dk = k.grad.data # Clipping
            dk.clamp_(-.1, 1.) # Multiply for lr
            dk.mul_(-lr) # Update
            k.data.copy_(k - dk)
        self.product.setCurvatures(K)
        
def learning(opt, G, epochs, curvaturesDetach = False, loss = 'default'):
    if loss == 'default': loss = defaultLoss
    losses = []; 
    X = opt.param_groups[0]["params"]
    K = opt.param_groups[1]["params"]
    if plot: manifolds.plot(X, opt.product)
    for epoch in tqdm(range(epochs)):
        opt.zero_grad()
        l = loss(X, G, opt.product); losses.append(l.data)
        l.backward(inputs = X + K)
        opt.step()
        if plot: manifolds.plot(X, opt.product)
    return X, K, losses

def lossij(X, G, P, i, j):
    dPij = P.distance(X[i], X[j])
    return ((dPij/(G[i,j].clamp_min(1e-15)))**2-1) ** 2

def defaultLoss(X : 'list tensors in P', 
                G : 'matrix of distances', 
                P : 'product manifold'):
    N = len(X); l = 0
    for i in range(N):
        for j in range(i+1, N):
            l += lossij(X, G, P, i, j)
    den = N**2 - N
    return l / den
                       