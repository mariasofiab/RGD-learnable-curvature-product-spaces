'''
Classes of manifold
'''
import torch
from torch import sqrt, cos, sin, cosh, sinh, arccos, tanh, atanh, arccosh, pi
from torch import tensor as tn
from matplotlib import pyplot as plt

def J(n):
    J = torch.eye(n); J[0,0] = -1
    return J

def dot(M):
    def f(u, v):
        partial = torch.matmul(M, v)
        return u.dot(v)
    return f

def mat(g, p, dim):
    M = torch.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            u = torch.eye(dim)[i]
            M[i,j] = torch.clamp(g(p, u, u),1e-15, 1e15)
    return M

def lorentzNorm(u):
    d = len(u)
    return sqrt((dot(J(d)))(u,u))

class Manifold():
    def __init__(self, dim : int, 
                 ambientDim : int, 
                 support : 'function that return True iff x belongs to Manifold', 
                 metricTensor : 'function of p, u, v',
                 curvature : torch.tensor
                 ):
        self.dim = int(dim)
        self.ambientDim = int(ambientDim)
        self.support = support # Function(x) that return true iff x belongs to manifold
        self.metricTensor = metricTensor # Function(p, u, v)
        self.curvature = tn(curvature, dtype = torch.float64, requires_grad = True)
        
    def g(self, p : "in M", u : "in TpM", v : "in TpM"):
        return self.metricTensor(p, u, v)
    
    def projection(self, p : 'in M', h : 'in Ambient Space'):
        if self.dim == self.ambientDim: return h
        return h - self.g(p, p, h) / self.g(p, p, p) * p
    
    def inverseTensor(self, p, u = None, v = None):
        M = mat(self.metricTensor, p, self.ambientDim)
        return 1 / M
        
    def __contains__(self, x : torch.tensor):
        if len(x) != self.dim: return False
        return self.support(x)
    
    def setCurvature(self, k):
        #old_k = self.curvature
        #factor = k / old_k
        self.curvature = k
        # distance = self.distance
        # def dist(p, q):
        #     return sqrt(1/factor) * distance(p, q)
        # self.distance = dist
        # metricTensor = self.metricTensor
        # self.metricTensor = lambda p, u, v: 1/factor * metricTensor(p, u, v) 
        
    def points2plottable(self, points):
        xs = []; ys = []; multiDim = len(points[0]) > 1
        for p in points:
            xs.append(p[0].data)
            if multiDim: ys.append(p[1].data)
            else: ys.append(tn(0.))
        return xs, ys
    
class EuclideanModel(Manifold):
    def __init__(self, dim):
        super().__init__(dim = dim, ambientDim = dim, 
                         support = lambda x: True, 
                         metricTensor = lambda p, u, v: u.dot(v),
                         #distance = dist,#lambda p, q, k: (p-q).norm(),
                         expMap = lambda p, v: v,
                         curvature = tn(0.))
        
    def distance(self, p, q):
        return (p-q).norm()
        
    def randomPoints(self, n = 1):
        res = [(torch.rand(self.ambientDim, dtype = torch.float64)-.5)*4 for i in range(n)]
        return res
    
    def __str__(self):
        return 'E^{%d}'%self.dim
    
class SphericModel(Manifold):
    def __init__(self, dim, curvature):
        curvature = abs(tn(curvature, dtype = torch.float64, requires_grad = True))
        super().__init__(dim = dim, ambientDim = dim+1, 
                         support = lambda x: abs(x.norm() - 1) < 1e-6, 
                         metricTensor = lambda p, u, v: (dot(torch.eye(dim+1)/curvature))(u, v),
                         curvature = curvature)

    def expMap(self, p, v):
        res = cos(v.norm())*p + sin(v.norm())  * v / v.norm()
        res /= res.norm()
        return res
        
    def distance(self, p, q):
        return arccos(p.dot(q).clamp(-1+1e-12,1-1e-12)) / sqrt(self.curvature).clamp_min(1e-15)
        
    def randomPoints(self, n = 1):
        res = [(torch.rand(self.ambientDim, dtype = torch.float64)-.5) for i in range(n)]
        res = [x / x.norm() for x in res]
        return res
    
    def __str__(self):
        return 'S^{%d}_{%.1f}'%(self.dim, self.curvature)
        
class LorentzModel(Manifold):
    def __init__(self, dim, curvature):
        curvature = -abs(tn(curvature, dtype = torch.float64, requires_grad = True))
        def expMap(p, v):
            norma = v.norm()
            vP = cos(norma)*p
            if norma > 0: vP += sin(norma)*v/norma
            vP /= vP.norm() # STABILITA' (NECESSARIO!!)
            return vP
        super().__init__(dim = dim, ambientDim = dim+1, 
                         support = lambda x: abs(lorentzNorm(x) + 1) < 1e-6, 
                         metricTensor = lambda p, u, v: (dot(J(dim+1)/curvature))(u, v),
                         curvature = curvature)
            
    def expMap(self, p, v):
        res = cosh(lorentzNorm(v))*p + sin(lorentzNorm(v))  * v / lorentzNorm(v)
        return res
        
    def distance(self, p, q):
        return arccosh(-(dot(J(self.ambientDim)))(p, q).clamp_min(1+1e-12)) / sqrt(self.curvature).clamp_min(1e-15)
        
    def randomPoints(self, n = 1):
        res = [(torch.rand(self.ambientDim, dtype = torch.float64)-.5)*4 for i in range(n)]
        for x in res:
            x = sqrt(1 + x[1:].norm()**2)
        return res
    
    def __str__(self):
        return 'H^{%d}_{%.1f}'%(self.dim, self.curvature)
        
class PoincareModel(Manifold):
    def __init__(self, dim, curvature):
        curvature = -abs(tn(curvature, dtype = torch.float64, requires_grad = True))
        super().__init__(dim = dim, ambientDim = dim,
                         support = lambda x: x.norm() < 1 + 1e-6, 
                         metricTensor = lambda p, u, v: (dot(torch.eye(dim)*(2/(1-p.dot(p)))/curvature))(u, v),
                         curvature = curvature)
        
    def expMap(self, p, v):
        norma = v.norm()
        lp = 2 / (1 - torch.sum(p ** 2, dim=-1, keepdim=True))
        s = sinh(lp*norma)
        c = cosh(lp*norma)
        d = p.dot(v/norma)
        den = 1 + c * (lp-1) + lp*d*s
        vP = lp*(c+d*s) * p + s * v / norma
        vP /= den
        if vP.norm() >= 1: vP *= .99 / vP.norm() # Stabilità
        return vP
        
    def distance(self, p, q):
        arg = 1 + 2 * ((p-q).dot(p-q)) / ((1-p.dot(p))*(1 - q.dot(q)))
        return arccosh(arg.clamp_min(1 + 1e-15)) / sqrt(-self.curvature).clamp_min(1e-15)
        
    def randomPoints(self, n = 1):
        res = [(torch.rand(self.ambientDim, dtype = torch.float64)-.5)*2 for i in range(n)] 
        res = [x / x.norm() * torch.rand(1, dtype = torch.float64) for x in res]
        return res
        
    def __str__(self):
        return 'D^{%d}_{%.1f}'%(self.dim, self.curvature)
    
class Product():
    def __init__(self, factors : 'list of ManifoldConstantCurvature'):
        self.factors = factors
        self.dim = sum([manifold.dim for manifold in factors])
        self.ambientDim = sum([manifold.ambientDim for manifold in factors])
        
    def g(self, p, u, v):
        res = tn(0); i = 0
        for manifold in self.factors:
            j = i + manifold.ambientDim
            res += manifold.metricTensor(p[i:j], u[i:j], v[i:j])
            i = j
        return res
    
    def support(self, x):
        i = 0
        for manifold in self.factors:
            j = i + manifold.ambientDim
            if not manifold.support(x[i:j]): return False
            i = j
        return True
    
    def distance(self, p, q):
        res = tn(0.); i = 0
        for count, manifold in enumerate(self.factors):
            j = i + manifold.ambientDim
            dist = manifold.distance(p[i:j], q[i:j])
            res += dist**2
            #res += (p[i:j] - q[i:j]).norm() / sqrt(K[count])
            i = j
        return sqrt(res)
    
    def expMap(self, p, v):
        res = torch.zeros(len(p)); i = 0
        for manifold in self.factors:
            j = i + manifold.ambientDim
            res[i:j] = manifold.expMap(p[i:j], v[i:j])
            i = j
        return res
    
    def inverseTensor(self, p, u = None, v = None):
        M = torch.zeros((self.ambientDim, self.ambientDim)); i = 0
        for manifold in self.factors:
            j = i + manifold.ambientDim
            M[i:j, i:j] = manifold.inverseTensor(p[i:j])
            i = j
        return M
    
    def projection(self, p, h):
        res = torch.zeros(len(p)); i = 0
        for manifold in self.factors:
            j = i + manifold.ambientDim
            res[i:j] = manifold.projection(p[i:j], h[i:j])
            i = j
        return res
    
    def setCurvatures(self, K):
        for i in range(len(K)):
            self.factors[i].setCurvature(K[i])
            
    def randomPoints(self, n = 1, grad = True):
        res = []
        for l in range(n):
            x = torch.zeros(self.ambientDim); i = 0
            for manifold in self.factors:
                j = i + manifold.ambientDim
                x[i:j] = manifold.randomPoints()[0]
                i = j
            x.requires_grad = grad
            res.append(x)
        return res
    
def plot(X : 'list of tensors in P', P):
    plottablePoints = []; i = 0;
    for count, manifold in enumerate(P.factors):
        j = i + manifold.ambientDim
        xs, ys = manifold.points2plottable([x[i:j] for x in X])
        plottablePoints.append((xs, ys, '$'+str(manifold)+'$'))
        i = j
    for xs, ys, label in plottablePoints:
        plt.scatter(xs, ys, label = label)
    plt.legend()
    plt.show()
