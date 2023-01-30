'''Tests'''
import torchRSGDpoincare
from torchRSGDpoincare import *
from costruisciGrafo import generaGrafo

def generaInS1(N):
    pts = []
    for i in range(N):
        theta = tn((2 * pi / N) * i)
        pts.append(tn([cos(theta), sin(theta)]))
    G = torch.zeros(N,N)
    for i in range(N):
        p = pts[i]
        for j in range(N):
            if i != j:
                q = pts[j]
                G[i,j] = distS(p,q,tn(1))
    return G

def testGeneric():
    N = 20
    lr = 1e-4
    batchSize = 1
    momentum = True
    sampling = 1
    clipping = False
    #Genero punti in S1
    G = generaInS1(N)
    # Scelgo Iperparametri
    S = [1,2]
    H = [2]
    e = 2
    epochs = 200
    losses, params, curvs, problem = metricEmbedding(G, S, H, e, epochs = epochs, lr = lr,
                                            sampling = sampling, plot = True,
                                            clipping = clipping,
                                            batchSize = batchSize, momentum = momentum)
    printCurvatures(params)

    print('Sono stato %f s a creare i minibatch.'%tempoBatchCreation)
    # E' più veloce senza miniBatch, e manco poco!

def testS1conpiuSample(N, sampling = .4):
    # Genero punti in S1
    G = generaInS1(N)
    # Scelgo Iperparametri
    S = [1,1]
    H = [1]
    e = 2
    lr = 1e-3
    batchSize = None
    momentum = False
    scaling = True
    epochs = 600
    print('Confronto due sample del %d%% ugualmente inizializzati. (N = %d)\n'%(int(100*sampling), N))
    # Fisso il seed
    K, X = randomStart(S, H, e, N)
    #Primo Sampling
    losses1, params1, curvs, problem = metricEmbedding(G, S, H, e, epochs = epochs, lr = lr, 
                                       X=X, K=K, sampling = sampling, plot = True,
                                       batchSize = batchSize, momentum = momentum, scaling = scaling)
    printCurvatures(params1)
    # Secondo Sampling
    losses2, params2, curvs, problem = metricEmbedding(G, S, H, e, epochs = epochs, lr = lr, 
                                       X=X, K=K, sampling = sampling, plot = True,
                                       batchSize = batchSize, momentum = momentum)
    printCurvatures(params2)
    plt.plot(losses1, label='Primo Sample'); plt.plot(losses2, label='Secondo Sample')
    plt.legend(); plt.title('Loss su %d epochs'%epochs)
    return params2


def testAllSettings():
    '''Prova tutti i settings per vedere quando si incorre in problemi'''
    G = generaInS1(20)
    #G = G/2
    lrs = [1e-2, 1e-4, 1e-9]
    batchSizes = [1, None]
    momentums = [True, False]
    samplings = [1, .6, .9]
    clippings = [True, False]
    epochs = 5
    S = [2]; H = [3]; e = 1;
    problems = []
    iterazione = 0
    maxIter = len(lrs) * len(batchSizes) * len(samplings) * 2**3
    for lr in lrs:
        for batchSize in batchSizes:
            for momentum in momentums:
                for sampling in samplings:
                    for clip in clippings:
                        iterazione +=1; print('%d/%d'%(iterazione, maxIter))
                        _, _, _, problem = metricEmbedding(G, S, H, e, 
                                                           epochs = epochs, lr = lr,
                                                           sampling = sampling, 
                                                           plot = False, 
                                                           batchSize = batchSize, 
                                                           momentum = momentum, 
                                                           clipping = clip)
                        if problem: problems.append({'lr':lr, 'bs':batchSize, 
                                                     'momentum':momentum, 'sampling':sampling,
                                                     'clipping':clip})
    for problem in problems:
        print(problem)
    if problems == []: print('Non ci sono stati problemi')

def testArticolo(params = None, lr = 1e-2):
    nPunti = 9
    torchRSGDpoincare.esercizioArticolo = nPunti
    plt.clf()
    G = generaGrafo(nPunti)
    plt.clf()
    batchSize = None
    momentum = True 
    clipping = False
    sampling = 1
    # Scelgo Iperparametri
    S = [2]
    H = [2]
    e = 0
    epochs = 1000
    losses, params, curvs, problem = metricEmbedding(G, S, H, e, epochs = epochs, 
                                            lr = lr, sampling = sampling, 
                                            plot = True, batchSize = batchSize,
                                            clipping = clipping, momentum = momentum, 
                                            params = params, saveRate = 50)
    printCurvatures(params)
    name = 'resultParams.json'
    saveParams(params, name)
    return losses, params, curvs

params = None
#paramsLoaded = loadParams('resultParams.json') # Da commentare in caso di prima esecuzione
losses, params, curvatures = testArticolo(params)

