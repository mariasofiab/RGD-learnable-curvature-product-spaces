###############################################################################
#                                                                             #
#   G = torch.tensor([[distG(vi, vj) for i in range(N)] for j in range(N)])   #
#       represent the finite distance to embed                                #  
#                                                                             #
#   S = torch.tensor((s1, ..., sm))      sTot = m + sum(S)                    #
#   H = torch.tensor((h1, ..., hn))      hTot = n + sum(H)                    #
#                                                                             #
#   X = [torch.rand(sTot + hTot + e, requires_grad=True) for i in range(N)]   #
#      mwhere X[i] respects condition of belong to P                          #
#                                                                             #
#   K = torch.tensor(k1, ..., km, c1, ..., cn, requires_grad=True)  Curvature #
#                                                                             #
#   params = X + [K, S, H, Sind, Hind, Eind, G]                               #
#                                                                             #
#   così X = params[:N]                                                       #
#        K = params[N]                                                        #
#        G = params[-1]                                                       #
#                                                                             #
#   X[i][l] è una componente :                                                #
#                                                                             #
#        sferica      se  l in range(sTot)                                    #
#        iperbolica   se  l in range(sTot, sTot + hTot)                       #
#        euclidea se  se  l in range(sTot + hTot, sTot + hTot + e)            #
#                                                                             #
###############################################################################
#                                                                             #
#   AGGIORNAMENTI                                                             #
#                                                                             #
#  -Aggiunti decoratori                                                       #
#                                                                             #
###############################################################################

import torch
from torch import tensor as tn
from torch import sqrt, cos, sin, cosh, sinh, arccos, tanh, atanh, arccosh, pi
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import plotly
from time import time as now
import json

error = '\033[31m'
black = '\033[30m'
esercizioArticolo = False

debug = True
superDebug = False

def lambda_p(p):
    return 2 / (1 - torch.sum(p ** 2, dim=-1, keepdim=True))

def getIndex(S, H, e):
    m = len(S)
    mSum, nSum = m + sum(S), sum(H)
    Sind = tn([(i + sum(S[:i]), sum(S[:i+1]) + i + 1, i) for i in range(len(S))])
    Hind = tn([(mSum + sum(H[:i]), mSum + sum(H[:i+1]), i) for i in range(len(H))])
    Eind = tn([(mSum + nSum, mSum + nSum + e, 0)])
    return [Sind.type(torch.IntTensor), Hind.type(torch.IntTensor), Eind.type(torch.IntTensor)]

def distS(u, v, k):
    eps = 1e-12
    arg = u.dot(v).clamp(-1+eps,1-eps)
    d = arccos(arg) / sqrt(k).clamp_min(1e-15)
    return d

def distH(u, v, k):
    arg = 1 + 2 * ((u-v).dot(u-v)) / ((1-u.dot(u))*(1 - v.dot(v)))
    d = arccosh(arg.clamp_min(1 + 1e-15)) / sqrt(k).clamp_min(1e-15)
    return d

def dist(u, v):
    d = (u-v).norm()
    return d

def distP(params, i, j):
    N = len(params) - 7
    X = params[:N]
    K, S, _, Sind, Hind, Eind, _ = params[N:]
    dP = 0
    # Spheric
    for a, b, fact in Sind:
        dP += distS(X[i][a:b], X[j][a:b], K[fact]) ** 2
    # Hyperbolic
    for a, b, fact in Hind:
        dP += distH(X[i][a:b], X[j][a:b], K[len(S) + fact]) ** 2
    # Euclidean
    for a, b, fact in Eind:
        dP += dist(X[i][a:b],X[j][a:b]) ** 2
    dP = sqrt(dP)
    return dP

def Lossij(params, i, j):
    G = params[-1]
    dPij = distP(params, i, j)
    return ((dPij/(G[i,j]+1e-12))**2-1) ** 2

def Loss(params):
    N = len(params) - 7
    loss = 0
    for i in range(N):
        for j in range(i+1, N):
            loss += Lossij(params, i, j)
    den = N**2 - N
    return loss / den

@torch.jit.script
def riemannianGrad(K, Sind, Hind, x, dx, sphDim):
    # Spheric
    for i, roba in enumerate(Sind):
        a = roba[0]; b = roba[1]; fact = roba[2]
        dx.data[a:b] *= K[int(fact)]
    # Hyperbolic
    for i, roba in enumerate(Hind):
        a = roba[0]; b = roba[1]; fact = roba[2]
        p = x[a:b]
        dx.data[a:b] *= K[sphDim+int(fact)] / lambda_p(p)**2
    return dx

@torch.jit.script
def projP(Sind, p, h):
    # Spheric
    for i, roba in enumerate(Sind):
        a = roba[0]; b = roba[1]
        newH = h[a:b] - h[a:b].dot(p[a:b]) * p[a:b]
        h[a:b] = newH
    return h

@torch.jit.script
def expP(Sind, Hind, Eind, p, v, sphDim):
    totDim = len(p)
    vP = torch.rand(totDim, dtype = torch.float64)
    # Spheric
    for i, roba in enumerate(Sind):
        a = roba[0]; b = roba[1]#; fact = roba[2]
        arg = v[a:b]# * sqrt(K[int(fact)])
        norma = arg.norm()
        vP[a:b] = cos(norma)*p[a:b] 
        if norma > 0: vP[a:b] += sin(norma)*arg/norma
        vP[a:b] /= vP[a:b].norm() # STABILITA' (NECESSARIO!!)
    # Hyperbolic
    for i, roba in enumerate(Hind):
        a = roba[0]; b = roba[1]#; fact = roba[2]
        arg = v[a:b]# * K[sphDim+fint(fact)]
        norma = arg.norm()
        lp = lambda_p(p[a:b])
        s = sinh(lp*norma)
        c = cosh(lp*norma)
        d = p[a:b].dot(arg/norma)
        den = 1 + c * (lp-1) + lp*d*s
        vP[a:b] = lp*(c+d*s) * p[a:b] + s * arg / norma
        vP[a:b] /= den
        if vP[a:b].norm() >= 1: vP[a:b] *= .99 / vP[a:b].norm() # Stabilità
    # Euclidean
    for i, roba in enumerate(Eind):
        a = roba[0]; b = roba[1]
        vP[a:b] = p[a:b] - v[a:b]
    return vP

class RiemannianSGD(torch.optim.Optimizer):
    def __init__(self, params):
        super(RiemannianSGD, self).__init__(params, {})

    def step(self, learningRate=0.3, bs = None, momentum = False, momenti = None, 
             alpha = tn(.9), clipping = True, indiciCoinvolti = None):
        eps = tn(1e-9)
        params = self.param_groups[0]['params']
        N = len(params) - 7
        K = params[N]
        S = params[N+1]
        Sind = params[N+3]
        Hind = params[N+4]
        Eind = params[N+5]
        sphDim = tn(len(S))
        # Scorro i mini batches dei punti sulla varietà
        normPts = 0
        normCurv = 0
        for i in indiciCoinvolti:
            if i == N: # Curvature
                lr = .05  # RIVEDIIIIIII
                #lr = learningRate
                if momentum:
                    m = momenti[1]
                    m = alpha * m - lr * K.grad
                    dK = - m / lr
                    momenti[1] = m
                else:
                    dK = K.grad * lr
                normCurv += dK.norm()
                if clipping:
                    dK = torch.clip(dK, -.1, .1) # Contro Exploding Gradient
                newK = K - lr * dK
                newK.data.clamp_(eps)
                self.param_groups[0]['params'][i].data.copy_(newK)
            else: # Punto
                x = params[i]
                dx = params[i].grad
                if momentum:
                    dx = params[i].grad
                    m = momenti[0][i]
                    m = alpha * m - learningRate * dx
                    momenti[0][i] = m
                    dx = - m / learningRate
                # Riemann Gradient
                dx = riemannianGrad(K, Sind, Hind, x, dx, sphDim)
                # Proiezione
                dx = projP(Sind, x, dx)
                normPts += dx.norm()**2
                # Clipping
                dx.clamp_(-1., 1.)
                # Exponential Map
                newx = expP(Sind, Hind, Eind, x.data, -learningRate * dx, sphDim)
                # Update
                self.param_groups[0]['params'][i].data.copy_(newx)
        normPts = sqrt(tn(normPts))
        if superDebug:
            if normPts != 0:
                print('||GradPts|| = %8.8f'%normPts)
            if normCurv != 0:
                print('||GradCrv|| = %8.8f'%normCurv)
        return True
            
def metricEmbedding(G : torch.tensor = tn([[0]]), 
                    sphericDims : list = [1],
                    hyperbolicDims : list = [1],
                    euclideanDim : int = 1, 
                    epochs = 100, params = None, 
                    lr = 1e-3, sampling = 1.,
                    X = None, K = None, momentum = False,
                    batchSize = None, clipping = True,
                    plot = False, saveRate = 50):
    problem = False
    N = len(G)
    if params is None:
        S = tn(sphericDims)
        H = tn(hyperbolicDims)
        if K is None or X is None:
            X, K = randomStart(S, H, euclideanDim, N)
        else:
            K = K.clone().detach().requires_grad_(True)
        indexes = getIndex(S, H, euclideanDim)
        params = X + [K, S, H] + indexes + [G]  
    else:
        N = len(params) - 7
        X = params[:N]
        K, S, H, Sind, Hind, Eind, G = params[N:]
    if debug: distAnalysis(params)
    X = params[:N]
    K, S, H, Sind, Hind, Eind, G = params[N:]    
    # Plot
    if plot:
        plotIncipit(N, sampling, batchSize, lr, momentum, clipping, epochs, S, H, K, X, euclideanDim, G)
    # Sampling
    if sampling < 1:
        sample = torch.randperm(N)
        N = int(tn(N * sampling))
        sample = sample[:N]
        X = [X[i] for i in sample]
        G = torch.index_select(G, dim=0, index=sample)
        G = torch.index_select(G, dim=1, index=sample)
        params = X + params[-7:]
    losses = []; curvs = [];
    if momentum:
        mX = [torch.zeros(len(X[0])) for i in range(N)]
        mK = torch.zeros(K.size()) # Momenti Primi
        momenti = [mX, mK]
    else:
        momenti = None
    # Create batches over pairs of point, saved as a pairs of indexes
    coppieIndici = []
    for i in range(N):
        for j in range(i+1, N):
            coppieIndici.append([i,j])
    if batchSize is None: batchSize = int(0.5 * (N**2 - N)) # Non Stochastic
    batches = DataLoader(range(len(coppieIndici)), batch_size=batchSize, shuffle=True)
    batches = [[coppieIndici[i] for i in batch] for batch in batches]
    numBatches = len(batches)
    # RSGD
    lossRate = 1
    opt = RiemannianSGD(params)
    for epoch in tqdm(range(epochs), 'Epochs'):
        # if debug and plot:
        #     plt.clf()
        #     comment = '%d° epoch / %d'%(epoch+1,epochs)
        #     drawPoints(S, H, euclideanDim, X, K, [Sind, Hind, Eind], comment=comment, edges = True, G = G)
        for batchIndex, batch in enumerate(batches):
            opt.zero_grad()
            loss = 0; indiciCoinvolti = set()
            for i, j in batch:
                indiciCoinvolti = indiciCoinvolti.union({i, j})
                lossij = Lossij(params, i, j)
                loss += lossij
            indiciCoinvolti = [i for i in indiciCoinvolti]# + [N] # Scommentare se si vuole trattare la curvatura come tutti gli altri parametri, ovvero aggiornandolo in ogni batchSize
            inputs = [params[i] for i in indiciCoinvolti]
            loss.backward(inputs = inputs)
            problem = not opt.step(learningRate=lr, clipping = clipping,
                            momentum=momentum, momenti = momenti, indiciCoinvolti = indiciCoinvolti)
            # if superDebug:
            #     comment = '%d° epoch / %d  -  %d° batch / %d'%(epoch+1, epochs, batchIndex + 1, numBatches)
            #     #comment += '  -  $||\nabla_{pts}||=%.2f$  -  $||\nabla_{curv}||=%.2f$'%()
            #     drawPoints(S, H, euclideanDim, X, K, [Sind, Hind, Eind], comment = comment)
        else:
            loss = Loss(params)
            loss.backward(inputs = params[N])
            problem = not opt.step(learningRate=lr, clipping = clipping,
                            momentum=momentum, momenti = momenti, indiciCoinvolti = [N])
            if not epoch % lossRate:
                loss = Loss(params)
                losses.append(float(loss.data))
            curvs.append(list(params[N].data.clone()))
            if not epoch % saveRate and epoch:
                saveParams(params, 'resultParams.json')
                plt.clf()
                plt.plot(losses); plt.title('Loss su %d epochs. Ultimo valore: %.3f'%(len(losses), losses[-1])); 
                plt.savefig('resultLoss', dpi = 600)
                plt.clf()
                plotCurvs(curvs, save = True)
                plt.clf()
                drawPoints(S, H, euclideanDim, params[:N], params[N], [Sind, Hind, Eind], save = True, 
                           comment='Result after %d epoch'%epoch, edges = True, G = G, plot = plot)
            continue
        break
    saveParams(params, 'resultParams.json')
    plt.clf()
    plt.plot(losses); plt.title('Loss su %d epochs. Ultimo valore: %.3f'%(len(losses), losses[-1])); 
    plt.savefig('resultLoss', dpi = 600)
    plt.clf()
    plotCurvs(curvs, save = True)
    plt.clf()
    drawPoints(S, H, euclideanDim, params[:N], params[N], [Sind, Hind, Eind], save = True, 
               comment='Result after %d epoch'%epoch, edges = True, G = G, plot = plot)
    return losses, params, curvs, problem

def plotIncipit(N, sampling, batchSize, lr, momentum, clipping, epochs, S, H, K, X, euclideanDim, G):
    print(black+'-Embedding di %d punti.'%N)
    if sampling < 1: print('-Eseguito un sampling delle istanze al %d%%.'%int(sampling*100))
    if not batchSize is None: print('-RSGD con mini-batch di dimensione %d con un learning rate di %.E.'%(batchSize,lr))
    else: print('-RSGD con un learning rate di %.E.'%lr)
    if momentum: print("-Ottimizzo l'algoritmo usando Momentum.")
    if clipping: print("-Uso il Clipping al gradiente.")
    print('-Si tentano di eseguire %d epochs.'%epochs)
    print('-Tutte le curvature sono inizializzate ad essere unitarie.')
    print('-I punti sono stati inizializzati come in figura:')
    drawPoints(S, H, euclideanDim, X, K, edges = True, G = G)

def distAnalysis(params, plot = True):
    N = len(params) - 7
    X = params[:N]
    K, S, H, Sind, Hind, Eind, G = params[N:]
    e = Eind[0][1]-Eind[0][0]
    proportions = getProportions(S, H, e, X, indexes=[Sind, Hind, Eind], K=K)
    distances = {'s':proportions[0],'h':proportions[1],'e':proportions[2]}
    print(black+'Distanze medie sui singoli fattori:')
    print('\nSFERICI:')
    for _, _, i in Sind:
        print('\tS^%d_%.1f:\t%2d' % (S[i],K[i],distances['s'][i]),'%')
    print('\nIPERBOLICI:')
    for _, _, i in Hind:
        print('\tD^%d_%.1f:\t%2d' % (H[i],K[len(S)+i],distances['h'][i]),'%')
    print('\nEUCLIDEO:')
    for _, _, i in Eind:
        print('\tE^%d:\t\t%2d' % (len(X[0])-sum(S)-sum(H) - len(S) - len(H),distances['e'][i]),'%\n')
    return distances

def printCurvatures(params):
    N = len(params) - 7
    K, S, H, Sind, Hind, Eind, G = params[N:]
    print('\nCURVATURE:\n')
    for a, b, i in Sind:
        print('\t%3d° fattore sferico:     dim = %2d     curv = %.1f' % (i, S[i], K[i]))
    for a, b, i in Hind:
        print('\t%3d° fattore iperbolico:  dim = %2d     curv = %.1f\n' % (i, H[i], K[i+len(S)]))

def randomStart(S, H, e, N, randomCurv = False):
    if debug: print('Inizializzo punti e curvature...')
    if randomCurv: K = torch.rand(len(S)+len(H), requires_grad=True, dtype = torch.float64)
    else: K = torch.ones((len(S)+len(H)), requires_grad=True, dtype = torch.float64)
    X = getRandomX(S, H, e, N)
    return X, K

def getRandomPinH(d):
    x = torch.rand(d,dtype=torch.float64) - .5
    x[0] = sqrt(1 + x[1:].norm()**2)
    return x

def getRandomX(S, H, e, N):
    indexes = getIndex(S, H, e)
    sTot = sum(S) + len(S); hTot = sum(H) + len(H)
    X = [torch.rand(sTot + hTot + e, dtype = torch.float64) - .5 for i in range(N)]
    for x in X:
        # Spheric
        for a, b, fact in indexes[0]:
            x[a:b] /= x[a:b].norm()
        # Hyperbolic
        for a, b, fact in indexes[1]:
            x[a:b] *= 2
            x[a:b] /= x[a:b].norm()
            x[a:b] *= torch.rand(1, dtype = torch.float64).clamp_(0,.9)
        # Euclidean
        for a, b, fact in indexes[2]:
            pass
        x.requires_grad=True
    return X

def getProportions(S, H, e, X, indexes = None, K = None):
    if K is None: K = tn([1.]*(len(S)+len(H)))
    if indexes is None: indexes = getIndex(S, H, e)
    Sind, Hind, Eind = indexes
    N = len(X)
    distances = {'s':torch.zeros(len(S)),'h':torch.zeros(len(H)),'e':torch.zeros(1)}
    for i in range(N):
        for j in range(N):
            if i != j:
                x = X[i]; y = X[j]
                for a, b, fact in Sind:
                    distances['s'][fact] += distS(x[a:b],y[a:b],K[fact])
                for a, b, fact in Hind:
                    distances['h'][fact] += distH(x[a:b],y[a:b],K[len(S)+fact])
                for a, b, fact in Eind:
                    distances['e'][fact] += dist(x[a:b],y[a:b])
    tot = sum(distances['s']) + sum(distances['h']) + sum(distances['e'])
    distances['s'] *= 100 / tot
    distances['h'] *= 100 / tot
    distances['e'] *= 100 / tot
    return [distances['s'],distances['h'],distances['e']]
                
def plotCurvs(curvs, save = False):
    epochs = len(curvs)
    n = len(curvs[0])
    for curvatura in range(n):
        plt.plot([curvs[i][curvatura] for i in range(epochs)], label='$K_{%d}$'%curvatura)
    plt.legend(); plt.title('Curvature nelle varie iterazioni')
    if save:
        plt.savefig('resultCurvatures.png',dpi=600)
    else:
        plt.show()
    
def lorentz2poincare(v):
    return v[1:] / (v[0]+1)

def poincare2lorentz(v):
    x0 = tn([1+v.norm()**2])
    xOther = 2*v
    return torch.cat((x0,xOther))/(1-v.norm()**2)

def geodetic(p, q, n = 2):
    x1, y1 = p
    x2, y2 = q
    #d = distH(p, q, tn(1))
    xs = []; ys = []
    for i in range(n):
        #t = i * d / (n - 1)
        t = i / (n-1)
        x = x1 * (1 - t) + x2 * t
        y = y1 * (1 - t) + y2 * t
        xs.append(x)
        ys.append(y)
    return xs, ys
    
def drawPoints(S, H, e, pts, K, indexes = None, comment = '', 
               hyperMode = 'Poincare', save = False, edges = False, 
               G = None, plot = True):
    colours = list(mcolors.TABLEAU_COLORS)
    if indexes is None: 
        indexes = getIndex(S, H, e)
    Sind, Hind, Eind = indexes
    m, n = len(S), len(H)
    points = {fact:{'x':[],'y':[],'name':''} for fact in range(m+n+1)}
    pDef = ''; volte = 0
    theta = torch.linspace(0, 2*pi, 100)
    x = torch.cos(theta)
    y = torch.sin(theta)
    plt.plot(x, y, '-k', linewidth=1)
    for p in pts:
        # Spheric
        for a, b, fact in indexes[0]:
            if not volte:
                if not pDef == '': pDef += ' \\times '
                pDef += 'S^{%d}_{%.1f}'%(S[fact], K[fact])
            if b-a>=2:
                points[int(fact)]['name'] = '$S^{%d}_{%.1f}$'%(S[fact], K[fact])
                points[int(fact)]['x'].append(p.data[a])
                points[int(fact)]['y'].append(p.data[a+1])
        # Hyperblic
        for a, b, fact in indexes[1]:
            if not volte:
                if not pDef == '': pDef += ' \\times '
                pDef += 'D^{%d}_{%.1f}'%(H[fact], K[m+fact])
            if b-a>=1:
                v = poincare2lorentz(p.data[a:b]) if hyperMode == 'Lorentz' else p.data[a:b]
                space = 'H' if hyperMode == 'Lorentz' else 'D'
                points[m+int(fact)]['name'] = '$'+space+'^{%d}_{%.1f}$'%(H[fact], K[m+fact])
                points[m+int(fact)]['x'].append(v[0])
                y = 0 if b-a == 1 else v[1]
                points[m+int(fact)]['y'].append(y)
        # Euclidean
        for a, b, fact in indexes[2]:
            if not volte and e:
                if not pDef == '': pDef += ' \\times '
                pDef += '\mathbb{R}^{%d}'%e
            if b-a>=2:
                points[m+n+int(fact)]['name'] = '$\mathbb{R}^{%d}$'%(b-a)
                points[m+n+int(fact)]['x'].append(p.data[a])
                points[m+n+int(fact)]['y'].append(p.data[a+1])
        volte += 1
    if edges:
        for i in range(len(pts)):
            p = pts[i].detach()
            for j in range(i):
                q = pts[j].detach()
                dist = G[i, j]
                # Spheric
                for a, b, fact in indexes[0]:
                    if b-a > 0:
                        plt.plot([p[a], q[a]], [p[a+1], q[a+1]], color=colours[fact], linewidth=float((1/dist).clamp_(.2, 2.5)))
                # Hyperbolic
                for a, b, fact in indexes[1]:
                    if b-a > 0:
                        plt.plot([p[a], q[a]], [p[a+1], q[a+1]], color=colours[fact+m], linewidth=float((1/dist).clamp_(.2, 2.5)))
                # Euclidean
                for a, b, fact in indexes[2]:
                    if b-a > 0:
                        plt.plot([p[a], q[a]], [p[a+1], q[a+1]], color=colours[fact+n+m], linewidth=float((1/dist).clamp_(.2, 2.5)))
    for fact in points.keys():
        plt.scatter(points[fact]['x'], points[fact]['y'], edgecolor = 'black', label=points[fact]['name'], linewidth = .4)
    plt.legend()
    plt.title(r'$\mathcal{P} = '+pDef+'$')
    plt.xlabel(comment)
    if save:
        plt.savefig('resultEmbedding.png',dpi=600)
    if plot: plt.show()

def saveParams(params, name = 'params.json'):
    tensors = [{'data': p.tolist(), 'requires_grad': p.requires_grad} for p in params]   
    with open(name, 'w') as f:
        json.dump(tensors, f)
        
def loadParams(name = 'params.json'):
    with open(name, 'r') as f:
        tensors = json.load(f)
    params = [torch.tensor(t['data'], dtype=torch.float64, requires_grad=t['requires_grad']) for t in tensors]
    N = len(params) - 7
    for i in range(N+1,N+7):
        params[i] = params[i].type(torch.IntTensor)
    return params
    
############################ PROVA ESECUZONE ##################################

'''
PROBLEMI

1)  I Risultati dipendono fortemente dal punto iniziale. La Loss non è convessa
    ed ha molti(ssimi) minimi locali, con relativi bacini di attrazione.
    Se ho un minimo locale soddisfacente allora ne ho un'infinità continua 
    in quanto il gruppo delle isometrie di P ha cardinalità continua. Quando mi 
    muovo tra di essi in realtà non è un problema. Un problema grande è questo:
        
    <<L'idea era di imparare le curvature dei linguaggi con Bert per poi
      confrontarle ed usarle come numeri rappresentativi del linguaggio stesso.
      Avendo tutti questi bacini di attrazione il significato delle curvature 
      apprese perde...>>
          --> Spera di osservare che i vari minimi hanno curvature simili
    
2)  ERRORI MACCHINA. Rendono l'algoritmo estremamente instabile. Ho apportato
    alcuni accorgimenti ma non bastano (ma aiutano parecchio)
        --> Quando incorri in NaN, non perdi il lavoro, ma arresti e ritorni
    Ma: con learning rate sensati (non troppo piccoli) si incorre sempre subito 
    in NaN
    
3)  Nel problema della tesi N sarà dell'ordine di 10^4 in quanto rappresenta il 
    numero di parole di un linguaggio. Ad ora è computazionalmente impensabile
    la sua trattazione con questo codice. 
        --> Prova Sampling del dataset
        --> Prova Negative Sampling sulla loss su cui faccio Backward
    
5)  La mappa esponenziale iperbolica non tiene conto della curvatura e quella
    sferica ne tiene conto ma non so se nel modo corretto 
        -->(Calcoli con Maria Sofia)
        -->Usa proiezione Ortogonale (non mi sembra una buona soluzione)
        
6)  Dai plot si nota che spesso i fattori Iperbolici / Euclidei scappano 
    all'infinito. Immagino sia in corso un gradiente che esplode.
        --> Fix
    Ma: che motivo dovrebbe esserci? Se il grafo G non ha distanze esagerate
    non c'è motivo per cui alcuni punti debbano scappare, se non quando la 
    curvatura iperbolica non sia prossima allo zero.
        --> Nel contesto Eucideo NON dovrebbe succedere. In quello iperbolico
            imposta un limite per la curvatura, ex curv(H) > 1e-2

7)  La curvatura sezionale è definita solo su varietà di dimensione >= 2. Come 
    faccio allora con i fattori unidimensionali?
    
8)  Il lr ottimale sembra dipendere dal numero di punti (Perchéééééé?????))
    
TO DO
2)  Riproduci esperimenti base del paper
3)  Aggiungi raggio di iniettività
5)  Tenta nuove Loss
6)  Idea: Più N è grande più il minimo minore è minore degli altri minimi
8)  Nel partire random, capisci almeno la scale del random cercando, ad 
    esempio, osservando max(G) e min(G)
10) Calcola Everage Distortion
13) Pre Training: trova la giusta permutazione dei punti di X
14) Fix NaN
15) Se due punti sono uguali, cosa succede? Ha senso fare arccos(clip(u.dot(v)))?
17) Plotta la norma dei gradienti nelle iterazioni per capire prché va così lento
18) Plotta spostamenti dei punti in qualche maniera
19) NaN viene fuori da K.grad eseguendo testArticolo. Probabilmente vengono 
    tutti fuori avvicinandosi alle singolarità della loss, ad esempio quando 
    l'argomento degli arccos si avvina o a 1 o a -1.
20) Relazione tra alpha e lr in GD with momentum
22) Se curvatura iperbolica troppo piccola i punti fuggono all'infinito
23) Fai in modo che il minibatch operi solo sui punti, e solo successivamente 
    vengano aggiorn ate le curvature
24) Togli ciò che non serve in 
    <<
            N = len(params) - 7
            X = params[:N]
            K, S, H, Sind, Hind, Eind, G = params[N:]
    >>
25) Usa coefficenti diversi per clipping discriminando in base a:
        -N
        -bastchSize
        -Componente sferica / iperbolica / euclidea
26)  Nuova Idea contro fuga iperbolica: diminuisci le distanze su G!
27) Spiega perché non hai usato geoopt: lì non si possono ottimizzare le curvature.
28) Ad ogni epoch stampa un istogramma che indica la magnitudine del gradiente
    rispetto ad ogni componente
29) Aggiungi la possibilità di lanciare l'algoritmo con curvature fissate scelte dall'inizio
30) Rivedi save / load params
31) Rivedi combo momentum - batchSize (credo non funzioni)
'''