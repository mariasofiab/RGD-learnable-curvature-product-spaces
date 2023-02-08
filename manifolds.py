import torch
from torch import sqrt, cos, sin, cosh, sinh, arccos, tanh, atanh, arccosh, pi
from torch import tensor as tn
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

def quadratic(us,diagonals,vs):
    if len(us.size()) == 1:
        return us.dot(diagonals*vs)
    return torch.matmul(us,(diagonals*vs).T).diag() # Da ottimizzare nello spazio

def tensorialDot(pts, qts):
    return quadratic(pts, torch.ones(pts.size()), qts)

def lorentzNorm(u):
    return sqrt(lorentzDot(u,u))

class Manifold():
    def __init__(self, dim : int, 
                 ambientDim : int, 
                 metricTensor : 'diagonal of matrix of dimention of ambient Space', # Non e' la vera diagonale! La dipendenza da curvatura e da p qui non e' esposta
                 curvature : torch.tensor
                 ):
        self.dim = int(dim)
        self.ambientDim = int(ambientDim)
        self.metricTensor = metricTensor # matrix of dimention of ambient Space
        self.curvature = tn(float(curvature), dtype = torch.float64, requires_grad = True)
        
    def g(self, pts : "in M", us : "in TpM", vs : "in TpM"):
        raws = self.metricTensor.broadcast_to((len(pts), self.ambientDim)) / self.curvature
        return quadratic(us, raws, vs)
    
    def norm(self, pts, us): # Di vettori sul tangente
        return sqrt(self.g(pts, us, us))
    
    def distance(self, pts, qts):
        res = (pts-qts).norm(dim=1)
        if res.isnan().any():
            print('Doh, un NaN!')
        return res
    
    def projection(self, pts : 'tensor of points in M as raws', hs : 'in Ambient Space'):
        if self.dim == self.ambientDim: return hs
        mul = self.g(pts, pts, hs) / self.norm(pts, pts)**2
        return hs - (pts.T*mul).T
    
    def inverseTensor(self, pts = None):
        raw = self.curvature / self.metricTensor.clamp_min_(1e-6)
        raws = raw.broadcast_to((len(pts), len(raw))) if not pts is None else raw
        return raws
    
    def randomPoints(self, n = 1):
        return torch.rand(n,self.ambientDim, dtype=torch.float64)-.5
    
    def expMap(self, pts, vs):
        res = pts + vs
        if res.isnan().any():
            print('Doh, un NaN!')
        return res
        
    def __contains__(self, x : torch.tensor):
        if len(x) != self.ambientDim: return False
        return True
        
    def points2plottable(self, points):
        n = len(points)
        if self.ambientDim == 1:
            return torch.cat((points, torch.zeros(n, 1)), dim=1).detach()
        return points[:,:2].detach()
    
class EuclideanModel(Manifold):
    def __init__(self, dim, curvature = 0):
        super().__init__(dim = dim, ambientDim = dim, 
                         metricTensor = torch.ones(dim),
                         curvature = tn(0.))
        
    def inverseTensor(self, pts = None):
        raw = 1 / self.metricTensor.clamp_min_(1e-6)
        raws = raw.broadcast_to((len(pts), len(raw))) if not pts is None else raw
        return raws
    
    def __str__(self):
        return 'E^{%d}'%self.dim
    
class SphericModel(Manifold):
    def __init__(self, dim, curvature):
        curvature = abs(tn(curvature, dtype = torch.float64, requires_grad = True))
        super().__init__(dim = dim, ambientDim = dim+1, 
                         metricTensor = torch.ones(dim+1),
                         curvature = curvature)

    def expMap(self, pts, vs):
        norms = torch.norm(vs, dim=1)
        res = (pts.T*cos(norms)).T + (vs.T*sin(norms)/norms.clamp_min(1e-12)).T
        res = (res.T/res.norm(dim=1)).T # Stabilita'!
        if res.isnan().any():
            print('Doh, un NaN!')
        return res
        
    def distance(self, pts, qts):
        eps = 1e-9
        res = arccos(tensorialDot(pts, qts).clamp(-1+eps,1-eps)) / sqrt(self.curvature).clamp_min(eps)
        if res.isnan().any():
            print('Doh, un NaN!')
        return res
        
    def randomPoints(self, n = 1):
        pts = super(SphericModel, self).randomPoints(n)
        pts = (pts.T/torch.norm(pts, dim=1)).T
        return pts
    
    def __contains__(self, x : torch.tensor):
        if len(x) != self.ambientDim: return False
        return abs(x.norm() - 1) < 1e-6
    
    def __str__(self):
        return 'S^{%d}_{%.1f}'%(self.dim, self.curvature.data)
        
class PoincareModel(Manifold):
    def __init__(self, dim, curvature):
        curvature = -abs(tn(curvature, dtype = torch.float64, requires_grad = True))
        super().__init__(dim = dim, ambientDim = dim,
                         metricTensor = torch.ones(dim),
                         curvature = curvature)
        
    def g(self, pts, us, vs):
        dots = super(PoincareModel, self).g(pts, us, vs)
        norms = pts.norm(dim=1)
        lps = 2 / (1 - norms**2).clamp_min_(1e-12)
        return dots*lps**2
        
    def expMap(self, pts, vs):
        norms = torch.norm(vs, dim=1)
        lps = 2 / (1 - torch.norm(pts, dim=1)**2).clamp_min_(1e-12)
        vsNormal = (vs.T/norms).T
        s = sinh((lps * norms).clamp_max(85)) # clamp per stabilità
        c = cosh((lps * norms).clamp_max(85)) # clamp per stabilità
        d = tensorialDot(pts, vsNormal)
        den = 1 + (lps - 1) * c + lps * d * s
        res = (pts.T*lps*(c + d * s)/den.clamp_min(1e-12)).T + (vsNormal.T*s/den.clamp_min(1e-12)).T
        eps = 1e-3
        res = (res.T/((1+eps)*res.norm(dim=1)/res.norm(dim=1).clamp_max(1)-eps)).T # Stabilita'
        if res.isnan().any():
            print('Doh, un NaN!')
        return res
    
    def inverseTensor(self, pts):
        raws = super(PoincareModel, self).inverseTensor(pts) * (-1) # curvatura negativa!
        norms = pts.norm(dim=1)
        lps = 2 / (1 - norms**2).clamp_min_(1e-12)
        return (raws.T/lps**2).T
        
    def distance(self, pts, qts):
        eps = 1e-9
        pqs = super(PoincareModel, self).distance(pts, qts) ** 2
        pps = tensorialDot(pts, pts).clamp_min(eps)
        qqs = tensorialDot(qts, qts).clamp_min(eps)
        arg = 1 + 2 * pqs / ((1-pps)*(1-qqs))
        res = arccosh(arg.clamp_min(1 + eps)) / sqrt(-self.curvature).clamp_min(eps)
        if res.isnan().any():
            print('Doh, un NaN!')
        return res
        
    def randomPoints(self, n = 1):
        pts = super(PoincareModel, self).randomPoints(n)
        pts = (pts.T/torch.norm(pts, dim=1)).T
        pts = (pts.T*torch.rand(n)*.99).T
        return pts
    
    def __contains__(self, x : torch.tensor):
        if len(x) != self.ambientDim: return False
        return x.norm() < 1 + 1e-6
        
    def __str__(self):
        return 'D^{%d}_{%.1f}'%(self.dim, self.curvature.data)
    
class Product():
    def __init__(self, factors : 'list of Manifold'):
        self.factors = factors
        self.dim = sum([manifold.dim for manifold in factors])
        self.ambientDim = sum([manifold.ambientDim for manifold in factors])
    
    def getCurvatures(self):
        return [M.curvature for M in self.factors]
        
    def g(self, pts, us, vs):
        i = self.factors[0].ambientDim
        res = self.factors[0].g(pts[:,:i], us[:,:i], vs[:,:i])
        for count, manifold in enumerate(self.factors):
            if count != 0:
                j = i + manifold.ambientDim
                res += manifold.g(pts[:,i:j], us[:,i:j], vs[:,i:j])
                i = j
        return res
    
    def support(self, x):
        i = 0
        for manifold in self.factors:
            j = i + manifold.ambientDim
            if not manifold.support(x[i:j]): return False
            i = j
        return True
    
    def distance(self, pts, qts):
        if len(pts.size())==1:
            pts = pts.broadcast_to((1,len(pts)))
            qts = qts.broadcast_to((1,len(qts)))
        i = self.factors[0].ambientDim
        res = self.factors[0].distance(pts[:,:i], qts[:,:i])**2
        for count, manifold in enumerate(self.factors):
            if count != 0:
                j = i + manifold.ambientDim
                res += manifold.distance(pts[:,i:j], qts[:,i:j])**2
                i = j
        return sqrt(res)
    
    def expMap(self, pts, vs):
        i = self.factors[0].ambientDim
        columns = self.factors[0].expMap(pts[:,:i],vs[:,:i])
        for count, manifold in enumerate(self.factors):
            if count != 0:
                j = i + manifold.ambientDim
                columns = torch.cat((columns, manifold.expMap(pts[:,i:j], vs[:,i:j])), dim=1)
                i = j
        if columns.isnan().any():
            print('Doh, un NaN!')
        return columns
    
    def inverseTensor(self, pts, u = None, v = None):
        i = self.factors[0].ambientDim
        columns = self.factors[0].inverseTensor(pts[:,:i])
        for count, manifold in enumerate(self.factors):
            if count != 0:
                j = i + manifold.ambientDim
                columns = torch.cat((columns, manifold.inverseTensor(pts[:,i:j])), dim=1)
                i = j
        return columns
    
    def projection(self, p, h):
        i = self.factors[0].ambientDim
        columns = self.factors[0].projection(p[:,:i],h[:,:i])
        for count, manifold in enumerate(self.factors):
            if count != 0:
                j = i + manifold.ambientDim
                columns = torch.cat((columns, manifold.projection(p[:,i:j], h[:,i:j])), dim=1)
                i = j
        return columns
            
    def randomPoints(self, n = 1):
        X = self.factors[0].randomPoints(n)
        for count, manifold in enumerate(self.factors):
            if count != 0:
                X = torch.cat((X,manifold.randomPoints(n)),dim=1)
        X.requires_grad_(True)
        return X
    
    def __str__(self):
        txt = str(self.factors[0])
        for i in range(1, len(self.factors)):
            txt += ' x ' + str(self.factors[i])
        return txt
    
def plotOnProduct(X, P, G = None):
    N = len(X)
    colours = list(mcolors.TABLEAU_COLORS)
    plottablePoints = []; i = 0;
    for count, manifold in enumerate(P.factors):
        j = i + manifold.ambientDim
        pts = manifold.points2plottable(X[:,i:j])
        plottablePoints.append((pts, '$'+str(manifold)+'$'))
        i = j
    # Plot edges
    if not G is None:
        vMax = G.max()
        vMin = (G+vMax*torch.eye(len(G))).min()
        deltaG = vMax-vMin
        for count, other in enumerate(plottablePoints): #Scorro sui fattori, non sui punti
            pts, _ = other
            color = colours[count]
            for i in range(N):
                for j in range(i):
                    width = float(.1 + .5 * (vMax-G[i, j])/deltaG)
                    plt.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]], color = color, linewidth=width)
    # Plot Points
    for pts, label in plottablePoints:
        plt.scatter(pts.data[:,0], pts.data[:,1], label = label)
    # Plot Circle
    theta = torch.linspace(0, 2*pi, 100)
    x = torch.cos(theta)
    y = torch.sin(theta)
    plt.plot(x, y, '-k', linewidth=1.)
    plt.legend()
    plt.show()