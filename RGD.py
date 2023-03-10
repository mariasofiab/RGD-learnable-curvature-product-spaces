import torch
from torch import tensor as tn
from torch import sqrt, cos, sin, cosh, sinh, arccos, tanh, atanh, arccosh, pi, exp
from tqdm import tqdm
from manifolds import *
import json
import os
import matplotlib.colors as mcolors

# plotOption:
# 0 non plot
# 1 plot only points every plotRate epochs
# 2 plot also edges every plotRate epochs
# 3 plot only loss AND curvatures
# 3.1 plot plot loss AND curvatures AND only points every plotRate epochs
# 3.2 plot plot loss AND curvatures AND points with edges every plotRate epochs
'''
X is tensor s.t. X[i,:] is the i° point and X[i, j] his j° coordinate.
K is a list of 1-tensors and K[i] is the magnitude of curvature of i° manifold.
'''

def riemannianGrad(X: torch.Tensor, product: Product):   
    gradient = X.grad.data
    gradient = product.projection(X, gradient)
    M = product.inverseTensor(X)
    return M*gradient

def computeLr(opt, epoch = None):
    lrX = opt.param_groups[0]["lr"] 
    lrK = opt.param_groups[1]["lr"]
    fact = 1
    if not epoch is None and not (1<=opt.drop<=1):
        fact = torch.pow(opt.drop, torch.floor(tn(epoch+1) / opt.epochsDrop))
    return lrX * fact, lrK * fact

class RSGD(torch.optim.Optimizer):
    def __init__(self, 
                 product : Product, 
                 numPoints = None, 
                 lr = 1e-2, 
                 X = None,
                 regolarization : 'in ["l1","l2"]' = None,
                 plotRate = 100,
                 plotOption = 3,
                 clipRange = 1, # torch.inf per non fare il clipping
                 epsCurv = 1e-6,
                 beta1 = .9, # Adam
                 beta2 = .99, # Adam
                 lReg = .1,
                 drop = 1,
                 epochsDrop = 100,
                 reduceLrOnPlateauStart = 0,
                 reduceLrOnPlateauFactor = 1.,
                 reduceLrOnPlateauThreshold = 1e-4
                 ): 
        if X is None and numPoints is None: raise ValueError('Devi fornire o X o numPoints')
        if X is None: X = product.randomPoints(n = numPoints)
        if numPoints is None: numPoints is len(X)
        K = product.getCurvatures()
        params = [
                    {"params": [X], "lr": lr},
                    {"params": K, "lr": 1e1 * lr / numPoints}
                 ]
        super(RSGD, self).__init__(params, {})
        self.product= product
        self.regolarization = regolarization
        self.plotRate = int(plotRate)
        self.plotOption = plotOption
        self.clipRange = tn(float(clipRange), device=device)
        self.epsCurv = tn(float(epsCurv), device=device)
        self.beta1 = tn(float(beta1), device=device)
        self.beta2 = tn(float(beta2), device=device)
        self.l = lReg
        self.drop = tn(float(drop))
        self.epochsDrop = tn(int(epochsDrop))
        self.reduceLrOnPlateauStart = reduceLrOnPlateauStart
        self.reduceLrOnPlateauFactor = reduceLrOnPlateauFactor
        self.reduceLrOnPlateauThreshold = reduceLrOnPlateauThreshold

    def step(self, losses, vX = None, vK = None, sX = None, sK = None, 
             epoch = None, startLearningCurvatures = 0):
        # Points
        X = self.param_groups[0]["params"][0]
        if len(losses) >= 2 and epoch >= self.reduceLrOnPlateauStart:
            if losses[-2] == 0 or losses[-1] == 0: return False, trn(0.), tn(0.)
            if self.reduceLrOnPlateauFactor != 1:
                if losses[-1] / losses[-2] >= 1 + self.reduceLrOnPlateauThreshold:
                    self.param_groups[0]["lr"] *= self.reduceLrOnPlateauFactor
                    self.param_groups[1]["lr"] *= self.reduceLrOnPlateauFactor
        lrX, lrK = computeLr(self, epoch)
        if not self.regolarization is None: 
            if self.regolarization == 'l1':
                f = regl1
            elif self.regolarization == 'l2':
                f = regl2
            lossReg = regl2(X, self.l)
            lossReg.backward()
        dX = riemannianGrad(X, self.product) # Riemannian Gradient
        dX.clamp_(-self.clipRange, self.clipRange)# Clipping
        if not vX is None: # Adam
            vX = self.beta1 * vX + (1 - self.beta1) * dX
            sX = self.beta2 * sX + (1 - self.beta2) * dX**2
            if self.beta2 == 1: dX = vX # momento
            else: dX = vX / ((1-self.beta1**(epoch+1))*sqrt(sX)/sqrt((1-self.beta2**(epoch+1)))+1e-8)
            if dX.isnan().any():
                dX = riemannianGrad(X, self.product)
                print('NaN')
        newX = self.product.expMap(X, -lrX*dX) # Exponential Map
        if newX.isnan().any():
            newX = self.product.expMap(X, -lrX*dX)
            print('NaN')
        X.data.copy_(newX) # Update
        # Curvatures
        K = self.param_groups[1]["params"]
        # Non occorre tensorializzare, tanto tipicamente sono poche componenti
        dKnorm = tn(0., device=device)
        if epoch >= startLearningCurvatures:
            for i in range(len(K)):
                k = K[i]
                if not k.grad is None:
                    dk = k.grad.data
                    dk.clamp_(-self.clipRange/lrK, self.clipRange/lrK) # Clipping
                    if not vK is None: # Adam
                        vK[i] = self.beta1 * vK[i] + (1 - self.beta1) * dk
                        sK[i] = self.beta2 * sK[i] + (1 - self.beta2) * dk**2
                        if self.beta2 == 1: dk = vK[i] # Momento
                        else: dk = vK[i] / ((1-self.beta1**(epoch+1))*sqrt(sK[i]/(1-self.beta2**(epoch+1)))+1e-8)
                        if dX.isnan().any():
                            print('NaN')
                    dKnorm += dk.norm()**2
                    if torch.abs(k) < 1:
                        dk *= sqrt(2*abs(k)-k**2) # Regolarizzo per evitare la divergenza in zero della derivata della curvatura
                    if k > 0:
                        newk = (k - lrK * dk).clamp_min(self.epsCurv)
                    else:
                        newk = (k - lrK * dk).clamp_max(self.epsCurv)
                    #newk = k.sign()*torch.max(self.epsCurv, torch.abs(k - lrK * dk)) # Distante epsCurv da 0
                    k.data.copy_(newk) # Update 
        dXavg = dX.norm() / sqrt(tn(dX.numel()))
        dKavg = sqrt(dKnorm / float(len(K)))
        continua = True
        if (dXavg < 1e-6 and dKavg < 1e-6) or lrX * lrK < 1e-16:
            if self.plotOption >= 3: print('\ndX average norm: %E\ndK average norm: %E'%(dXavg, dKavg))
            continua = False # Early Exit
        return continua, dXavg, dKavg
        
def learning(opt, G, epochs, loss = 'articolo',
             no_curv = False, adam = False, startLearningCurvatures = 0, 
             savePlt = False, saveName = ''):
    if not loss in lossChoices.keys(): raise ValueError(str(loss)+" doesn't is implemented")
    if not saveName == '':
        if not os.path.exists('Immagini/'+saveName):
            os.makedirs('Immagini/'+saveName)
        saveName += '/'
    G = G.to(device)
    losses = [] 
    X = opt.param_groups[0]["params"][0]
    K = opt.param_groups[1]["params"]
    if opt.plotOption: plotIncipit(opt, len(G), epochs, no_curv, adam, 
                                   averageDistortion(X, opt.product, G), 
                                   loss, startLearningCurvatures, savePlt, saveName)
    loss = lossChoices[loss]
    comment = 'Epoch: %d - Davg: %.2f'%(0, averageDistortion(X, opt.product, G))
    if opt.plotOption in (1, 3.1): plotOnProduct(X, opt.product, comment = comment, savePlt = savePlt, saveName = saveName+'start') 
    elif opt.plotOption in (2, 3.2): plotOnProduct(X, opt.product, G = G, comment = comment, savePlt = savePlt, saveName = saveName+'start')
    curvs = [[float(k) for k in K]]
    if adam: 
        # Momenti primi
        dX = torch.zeros(X.size(), dtype = torch.float64, device = device)
        dK = [tn(0., dtype = torch.float64, device = device) for k in K]
        # Momenti secondi
        ddX = torch.zeros(X.size(), dtype = torch.float64, device = device)
        ddK = [tn(0., dtype = torch.float64, device = device) for k in K]
    else:
        dX = None; dK = None; ddX = None; ddK = None
    # Ciclo sulle EPOCHS
    iterabile = tqdm(range(epochs))
    #iterabile = range(epochs)
    for epoch in iterabile:
        opt.zero_grad()
        l = loss(X, G, opt.product)
        if l.isnan():
            print('NaN')
        if int(opt.plotOption) == 3: losses.append(float(l.data))
        else:
            if len(losses) <= 1:
                losses.append(l)
            else:
                losses[0] = losses[1]; losses[1] = l
        if no_curv:
            l.backward(inputs = [X])
        else:
            l.backward(inputs = [X] + K)
        continua, dXavg, dKavg = opt.step(losses, dX, dK, ddX, ddK, epoch = epoch, startLearningCurvatures = startLearningCurvatures)
        if X.isnan().any():
            raise FloatingPointError("NaN")
            break
        curvs.append([float(k) for k in K])
        if not epoch % opt.plotRate and epoch != 0:
            davg = averageDistortion(X, opt.product, G)
            comment = 'Epoch: %d - Davg: %.2f'%(epoch, davg)
            if opt.plotOption  in (1, 3.1): plotOnProduct(X, opt.product, comment = comment) 
            elif opt.plotOption  in (2, 3.2): plotOnProduct(X, opt.product, G = G, comment = comment)
        if not continua: break
    davg = averageDistortion(X, opt.product, G)
    if opt.plotOption and savePlt:
        comment = 'Epoch: %d -  Loss: %.2f  - Davg: %.2f'%(epoch, l, davg)
        if opt.plotOption  in (1, 3.1): plotOnProduct(X, opt.product, comment = comment, savePlt = savePlt, saveName = saveName) 
        elif opt.plotOption  in (2, 3.2): plotOnProduct(X, opt.product, G = G, comment = comment, savePlt = savePlt, saveName = saveName)
    if int(opt.plotOption) == 3:
        losses.append(float(loss(X, G, opt.product).data))
        plotResults(opt, losses, davg, dXavg, dKavg, curvs, X, savePlt = savePlt, saveName = saveName)
    return X, opt.product, davg

def regl1(X, l):
    return l * sum(X.norm(dim=1)) / len(X)

def regl2(X, l):
    return l * sum(X.norm(dim=1) ** 2) / len(X)

def loss1(X : 'tensor of points in P', 
          G : 'matrix of distances', 
          P : 'product manifold',
          I : 'list of point indexes' = None):
    N = len(X); l = 0
    if I is None: I = range(1,N)
    for i in I:
        diP = P.distance(X[i,:].broadcast_to(X[:i, :].size()), X[:i, :])
        l += torch.sum(((diP / G[i,:i])**2 - 1)**2)
        #l += torch.sum(torch.abs((diP / G[i,:i])**2 - 1))
    return l / (N-1) # Rivedi normalizzazione in caso I non sia tutto {0, ..., N-1}

def loss2(X : 'tensor of points in P', 
          G : 'matrix of distances', 
          P : 'product manifold',
          I : 'list of point indexes' = None):
    N = len(X); l = 0
    if I is None: I = range(1,N)
    for i in I:
        diP = P.distance(X[i,:].broadcast_to(X[:i, :].size()), X[:i, :])
        l += torch.sum((diP - G[i,:i])**2)
    return l / (N-1)

def loss3(X : 'tensor of points in P', 
          G : 'matrix of distances', 
          P : 'product manifold',
          I : 'list of point indexes' = None):
    #return (loss1(X, G, P, I)+loss2(X, G, P, I))*.5
    N = len(X); l = 0
    if I is None: I = range(1,N)
    for i in I:
        diP = P.distance(X[i,:].broadcast_to(X[:i, :].size()), X[:i, :])
        l += torch.sum(((diP / G[i,:i])**2 - 1)**2) + 4 * torch.sum((diP - G[i,:i])**2)
    return l / (2*(N-1))

lossChoices = {
    'articolo':loss1,
    'mse':loss2,
    'sum':loss3
    }

def averageDistortion(X, P, G):
    N = len(X); res = 0
    for i in range(N):
        diP = P.distance(X[i,:].broadcast_to(X[:i, :].size()), X[:i, :])
        res += sum(abs(G[i,:i] - diP)/G[i,:i].clamp_min(1e-9))
    return 2 * res / (N**2 - N)

def getIncipit(opt, N, epochs, no_curv, adam, davg, loss, startLearningCurvatures):
    txt = ''; txt2 = ''
    txt += 'P =        $'+str(opt.product)+'$\n'
    txt += 'Epochs:    $%d$\n'%epochs
    txt += 'X lr:      $%.1E$\n'%opt.param_groups[0]["lr"]
    txt += 'K lr:      $%.1E$\n'%opt.param_groups[1]["lr"]
    txt += '#Points:   $%d$\n'%N
    txt += 'loss:      $%s$\n'%loss
    if adam: 
        if opt.beta2 != 1:
            txt += 'Adam:      $%s$\n'%str(adam)
            txt += 'Beta1:     $%.1E$\n'%opt.beta1
            txt += 'Beta2:     $%.1E$\n'%opt.beta2
        else:
            txt += 'Momento:   $%s$\n'%str(adam)
            txt += 'Beta:      $%.1E$\n'%opt.beta1
    txt += 'Clipping:  $%s$\n'%str(bool(opt.clipRange != torch.inf))
    txt += 'No_curv:   $%s$\n'%str(no_curv)
    txt2 += 'Regolarization:        $%s$\n'%str(opt.regolarization)
    txt += 'min|Curv|: $%.1E$\n'%opt.epsCurv
    txt2 += 'Initial Davg:          $%.2f$\n'%davg
    if opt.clipRange != torch.inf: 
        txt2 += 'Clip threshold:        $%.1E$\n'%opt.clipRange
    if not opt.regolarization is None:
        txt += 'lambda:    $%.2f$\n'%opt.l
    if not (1<=opt.drop<=1):
        txt2 += 'Drop; epochsDrop       $%.2E; %d$\n'%(opt.drop, opt.epochsDrop)
    if startLearningCurvatures:
        txt2 += 'Learn Curvs from epoch $%d$\n'%startLearningCurvatures
    if opt.reduceLrOnPlateauFactor != 1:
        txt2 += 'reducePlateauStart     $%d$\n'%opt.reduceLrOnPlateauStart
        txt2 += 'reducePlateauFactor    $%.2E$\n'%opt.reduceLrOnPlateauFactor
        txt2 += 'reducePlateauThreshold $%.2E$\n'%opt.reduceLrOnPlateauThreshold
    return txt, txt2
    
def plotIncipit(opt, N, epochs, no_curv, adam, davg, loss, startLearningCurvatures, savePlt = False, saveName = ''):
    txt1, txt2 = getIncipit(opt, N, epochs, no_curv, adam, davg, loss, startLearningCurvatures)
    fig, ax = plt.subplots()
    ax.text(0.05,0.01,txt1, fontsize=9, fontfamily="monospace")
    ax.text(.4,0.01,txt2, fontsize=9, fontfamily="monospace")
    plt.title('Settings')
    ax.set_xticks([])
    ax.set_yticks([])
    if savePlt:
        plt.savefig('Immagini/'+saveName+'Settings.png',dpi=600)
    plt.show()
    
def plotResults(opt, losses, davg, dXavg, dKavg, curvs, X, savePlt = False, saveName = ''):
    # Percentage
    drawPercentages(X, opt.product, savePlt, saveName)
    # Curvastures
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for j in range(len(opt.product.factors)):
        fact = opt.product.factors[j]
        if not type(fact) is EuclideanModel:
            ax1.plot([curvs[i][j] for i in range(len(curvs))], label='$'+str(fact)+'$')
            ax2.plot(range(max(0,len(curvs)-100),len(curvs)), [curvs[i][j] for i in range(max(0,len(curvs)-100),len(curvs))], label='$'+str(fact)+'$')
    ax1.legend(); ax2.legend();
    fig.suptitle('Curvature nelle Epoch')
    if savePlt: plt.savefig('Immagini/'+saveName+'Curvatures.png',dpi=600)
    plt.show()
    # Losses
    fig, (ax1, ax2) = plt.subplots(1, 2)
    txt ='Last Loss: %.2f  -  $D_{avg}$: %.6f'%(losses[-1], davg)
    txt +='  -  $||dX||_{avg}$: %.0E  -  $||dK||_{avg}$: %.0E'%(dXavg, dKavg)
    fig.suptitle(txt)
    ax1.plot(torch.log2(tn(losses)), label='$log(Loss)$'); ax1.legend()
    ax1.plot(torch.log2(tn(losses)+1), label='$log(Loss+1)$'); ax1.legend()
    ax2.plot(losses[-100:], label='LastLosses'); ax2.legend()
    if savePlt: plt.savefig('Immagini/'+saveName+'Losses.png',dpi=600)
    plt.show()
    
def drawPercentages(X, P, savePlt = False, saveName = ''):
    colours = list(mcolors.TABLEAU_COLORS)
    bins = ['$'+str(M)+'$' for M in P.factors]
    heights = P.repartitionOfDistances(X)
    plt.bar(bins, heights, align='center', alpha=0.5, color=colours)
    plt.xticks(bins, bins)
    plt.ylabel('Percentage of Distance\nexplained by factors')
    if savePlt: 
        plt.savefig('Immagini/'+saveName+'Percentages.png',dpi=600)
    plt.show()
    
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
    X = tn(data['X'], dtype = torch.float64, requires_grad = True, device = device)
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
5.  Scala il learning rate delle curvature                                    V
6.  Rendi compatibile con la GPU il plot dei punti
7.  Aggiungi altri grafi                                                      V
8.  Aggiungi visualizzazione distance explaination                            V
9.  Aggiungi regolarizzazioni                                                 V (l1 e l2)
10. Cambia le variabili globali con parametri di opt                          V
11. Early Stop                                                                V
12. Dividi momento punti da momento curvature
13. Decay of Learning Rate                                                    V
14. Trova una funzione k(N, d) per cui l'attesa della distanza media di N punti
    generati in un fattore di dimensione d sia 1. Fai il conto separatamente 
    per fattore sferico / iperbolico.
15. Velocizza creando una sola volta dei tensori che durante il learning 
    andrebbero sempre rifatti
16. Sposta tutti i setting nell'init, e non nel training.
17. Adam!!                                                                    V
18. Salva grafici hd
19. Ora faccio prima il clipping e poi adam / momentum                        V
20. Aggiungi un termine di regolarizzazione sulla varianza dell'errore '
21. Quando salvi, salva anche i parametri!
'''
