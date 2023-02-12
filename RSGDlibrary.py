import torch
from torch import tensor as tn
from torch import sqrt, cos, sin, cosh, sinh, arccos, tanh, atanh, arccosh, pi
from tqdm import tqdm
from manifolds import *
import json
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU instead")

plotOption = 3
# 0 non plot
# 1 plot only points every epoch
# 2 plot also edges every epoch
# 3 plot only loss AND curvatures
# 3.1 plot plot loss AND curvatures AND only points every epoch
# 3.2 plot plot loss AND curvatures AND points with edges every epoch
clipRange = tn(1.) # torch.inf per non fare il clipping
epsCurv = tn(1e-6)
beta = tn(.5) # Per il momento
'''
X is tensor s.t. X[i,:] is the i° point and X[i, j] his j° coordinate.
K is a list of 1-tensors and K[i] is the magnitude of curvature of i° manifold.
'''

def riemannianGrad(X: torch.Tensor, product: Product):   
    gradient = X.grad.data
    if gradient.isnan().any():
        print('Doh, un NaN!') # Ok, lui è il primo a generare NaN
    M = product.inverseTensor(X)
    return M*gradient

class RSGD(torch.optim.Optimizer):
    def __init__(self, product : Product, numPoints = None, lr = 1e-2, X = None):
        if X is None and numPoints is None: raise ValueError('Devi fornire o X o numPoints')
        if X is None: X = product.randomPoints(n = numPoints).to(device)
        if numPoints is None: numPoints is len(X)
        K = product.getCurvatures()
        for k in K: k = k.to(device)
        params = [
                    {"params": [X], "lr": lr},
                    {"params": K, "lr": lr / numPoints}
                 ]
        super(RSGD, self).__init__(params, {})
        self.product= product

    def step(self, dXold = None, dKold = None):
        global clipRange, epsCurv, beta
        # Points
        X = self.param_groups[0]["params"][0]
        lr = self.param_groups[0]["lr"]
        dX = riemannianGrad(X, self.product) # Riemannian Gradient
        dX = self.product.projection(X, dX) # Projection
        dX.clamp_(-clipRange, clipRange)# Clipping
        if not dXold is None:
            dX = beta * dXold + (1 - beta) * dX # Momento
            dXold = dX
        newX = self.product.expMap(X, -lr*dX) # Exponential Map
        X.data.copy_(newX) # Update
        # Curvatures
        K = self.param_groups[1]["params"]
        # Non occorre tensorializzare, tanto tipicamente sono poche componenti
        lr = self.param_groups[1]["lr"]
        for i in range(len(K)):
            k = K[i]
            dk = k.grad.data
            if dk.isnan():
                print('Doh, un NaN!')
            dk.clamp_(-clipRange, clipRange) # Clipping
            if not dKold is None: 
                dk = beta * dKold[i] + (1 - beta) * dk # Momento
                dKold[i] = dk
            newk = k.sign()*torch.max(epsCurv, torch.abs(k - lr * dk)) # Distante epsCurv da 0
            k.data.copy_(newk) # Update 
        
def learning(opt, G, epochs, curvaturesDetach = False, loss = 'default',
             no_curv = False, momento = True):
    if plotOption: printIncipit(opt, len(G), epochs, no_curv, momento)
    if loss == 'default': loss = defaultLoss
    if int(plotOption) == 3: losses = [] 
    X = opt.param_groups[0]["params"][0]
    K = opt.param_groups[1]["params"]
    if plotOption in (1, 3.1): plotOnProduct(X, opt.product) 
    elif plotOption in (2, 3.2): plotOnProduct(X, opt.product, G = G)
    curvs = [[float(k) for k in K]]
    if momento: 
        dX = torch.zeros(X.size(), dtype = torch.float64)
        dK = [tn(0., dtype = torch.float64) for k in K]
    else:
        dX = None; dK = None
    for epoch in tqdm(range(epochs)):
        opt.zero_grad()
        l = loss(X, G, opt.product)
        if l.isnan().any():
            print('Doh, un NaN!')
        if int(plotOption) == 3: losses.append(float(l.data))
        if no_curv:
            l.backward(inputs = [X])
        else:
            l.backward(inputs = [X] + K)
        opt.step(dX, dK)
        curvs.append([float(k) for k in K])
        if plotOption  in (1, 3.1): plotOnProduct(X, opt.product) 
        elif plotOption  in (1, 3.2): plotOnProduct(X, opt.product, G = G)
    devg = everageDistortion(X, opt.product, G)
    if int(plotOption) == 3:
        losses.append(float(loss(X, G, opt.product).data))
        fig, ax = plt.subplots()
        ax.plot(losses)
        ax.set_title('$P='+str(opt.product)+'$\nLast Loss: %.2f. -  $D_{evg}$: %.2f'%(losses[-1], devg))
        plt.show()
        fig, ax = plt.subplots()
        for j in range(len(opt.product.factors)):
            fact = opt.product.factors[j]
            if not type(fact) is EuclideanModel:
                ax.plot([curvs[i][j] for i in range(len(curvs))], label='$'+str(fact)+'$')
        ax.set_title('Curvature nelle Epoch')
        plt.legend(); plt.show()
    return X, opt.product, devg

def defaultLoss(X : 'tensor of points in P', 
                G : 'matrix of distances', 
                P : 'product manifold'):
    N = len(X); l = 0
    for i in range(1,N): # Se dovessi aggiungere i minibatch li farei su questo for
        diP = P.distance(X[i,:].broadcast_to(X[:i, :].size()), X[:i, :])
        l += torch.sum(((diP / G[i,:i])**2 - 1)**2)
    return l / (N-1)

def everageDistortion(X, P, G):
    N = len(X); res = 0
    for i in range(N):
        diP = P.distance(X[i,:].broadcast_to(X[:i, :].size()), X[:i, :])
        res += sum(abs(G[i,:i] - diP)/G[i,:i].clamp_min(1e-9))
    return 2 * res / (N**2 - N)

def printIncipit(opt, N, epochs, no_curv, momento):
    global clipRange, epsCurv, beta
    print('SETTINGS:')
    print('\tEpochs:          %d'%epochs)
    print('\tX lr:            %E'%opt.param_groups[0]["lr"])
    print('\tK lr:            %E'%opt.param_groups[1]["lr"])
    print('\t#Points:         %d'%N)
    print('\tMomento:         %s'%str(momento))
    if momento: print('\tBeta:            %f'%beta)
    print('\tClipping:        %s'%str(clipRange != torch.inf))
    if clipRange != torch.inf: print('\tClip threshold:  %f'%clipRange)
    print('\tNo_curv:         %s'%str(no_curv))
    
def save(X, P, name = 'params.json', G = None):
    data = {'X': X.tolist(), 
            'K': [float(fact.curvature) for fact in P.factors],
            'Dims' : [fact.dim for fact in P.factors],
            'FactNames': [str(type(fact)) for fact in P.factors]}
    if not G is None: data['G'] = G.tolist()
    with open(name, 'w') as f:
        json.dump(data, f)    
        
def load(name = 'params.json'):
    with open(name, 'r') as f:
        data = json.load(f)
    X = tn(data['X'], dtype = torch.float64, requires_grad = True)
    K = data['K']
    dims = data['Dims']
    types = [getType(s) for s in data['FactNames']]
    P = Product([types[i](dims[i], K[i]) for i in range(len(types))])
    return X, P

def getType(s):
    s = s[:-2]
    sLast = ''
    for i in range(len(s) - 1, -1, -1):
        if s[i].isalpha():
            sLast = s[i] + sLast
        else:
            break
    return eval(sLast)


'''
TO DO:
1.  Momento                                                                   V
2.  Salva
3.  Grafica le curvature nelle epochs                                         V
4.  Mini Batches
'''