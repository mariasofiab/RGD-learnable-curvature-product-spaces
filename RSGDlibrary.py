import torch
from torch import tensor as tn
from torch import sqrt, cos, sin, cosh, sinh, arccos, tanh, atanh, arccosh, pi
import manifolds
from tqdm import tqdm

plot = 1 # 0 non plot, 1 plot only points, 2 plot also edges
clipRange = tn(1.) # torch.inf per non fare il clipping
epsCurv = tn(1e-6)
'''
X is tensor s.t. X[i,:] is the i° point and X[i, j] his j° coordinate.
K is a list of 1-tensors and K[i] is the magnitude of curvature of i° manifold.
'''

def riemannianGrad(X: torch.Tensor, product: manifolds.Product):   
    gradient = X.grad.data
    M = product.inverseTensor(X)
    return M*gradient

class RSGD(torch.optim.Optimizer):
    def __init__(self, product : manifolds.Product, numPoints, lr = 1e-2, X = None):
        if X is None: X = product.randomPoints(n = numPoints)
        K = [M.curvature for M in product.factors]
        params = [
                    {"params": [X], "lr": lr},
                    {"params": K, "lr": lr}
                 ]
        super(RSGD, self).__init__(params, {})
        self.product= product

    def step(self):
        global clipRange, epsCurv
        # Points
        X = self.param_groups[0]["params"][0]
        lr = self.param_groups[0]["lr"]
        if not X.grad is None:
            dX = riemannianGrad(X, self.product) # Riemannian Gradient
            dX = self.product.projection(X, dX) # Projection
            dX.clamp_(-clipRange, clipRange)# Clipping
            newX = self.product.expMap(X, -lr*dX) # Exponential Map
            X.data.copy_(newX) # Update
        # Curvatures
        K = self.param_groups[1]["params"]
        # Non occorre tensorializzare, tanto tipicamente sono poche componenti
        lr = self.param_groups[0]["lr"]
        for i in range(len(K)):
            k = K[i]
            if k.grad is None:
               continue
            dk = k.grad.data
            dk.clamp_(-clipRange, clipRange) # Clipping
            dk.mul_(-lr)  # Multiply for lr
            newk = k.sign()*torch.max(epsCurv, torch.abs(k - dk)) # Distante epsCurv da 0
            k.data.copy_(newk) # Update 
        
def learning(opt, G, epochs, curvaturesDetach = False, no_curv = False, loss = 'default'):
    if loss == 'default': loss = defaultLoss
    losses = []; 
    X = opt.param_groups[0]["params"][0]
    K = opt.param_groups[1]["params"]
    if plot == 1: manifolds.plot(X, opt.product) 
    elif plot == 2: manifolds.plot(X, opt.product, G = G)
    for epoch in tqdm(range(epochs)):
        opt.zero_grad()
        l = loss(X, G, opt.product); losses.append(l.data)
        if no_curv:
            l.backward(inputs = [X])
        else:
            l.backward(inputs = [X] + K)
        opt.step()
        if plot == 1: manifolds.plot(X, opt.product) 
        elif plot == 2: manifolds.plot(X, opt.product, G = G)
    return X, K, losses

def defaultLoss(X : 'tensor of points in P', 
                G : 'matrix of distances', 
                P : 'product manifold'):
    '''
    Ora è un solo for! 
    Non ho avuto un'idea buona per eliminare ange questo. L'idea è stata di 
    creare un 3-tensore di cui una buona metà degli elementi sono zero.
    In qualche modo credo che si possa fare, ma se il nostro problema è di
    spazio prima che di tempo, mi sembra una scelta cattiva.
    '''
    N = len(X); l = 0; addendi = 0;
    for i in range(1,N): # Se dovessi aggiungere i minibatch li farei su questo for
        diP = P.distance(X[i,:].broadcast_to(X[:i, :].size()), X[:i, :])
        l += torch.sum(((diP / G[i,:i])**2 - 1)**2)
        addendi += len(diP)
    return l / addendi
                       
'''To do:
1   dimezzo il lr quando la loss cresce
2   aggiungo batch
3   aggiungo momento    
4   tensorializza la loss! Fatto a metà
5   ottieni una loss che almeno diminuisca (se, magariii)
'''