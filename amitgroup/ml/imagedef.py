
import numpy as np
import amitgroup as ag
from aux import deform_x
from itertools import product
from math import cos

__all__ = ['imagedef', 'deform']

twopi = 2.0 * np.pi

def psi(k1, k2, x):
    return twopi * cos(twopi * (k1 * x[0] + k2 * x[1]))

def scriptN(a):
    return a 

# Use ag.ml.deform_x instead
def __deprecated_deform_x(xs, u, n, psis):
    zs = np.copy(xs) 
    for x0 in range(xs.shape[0]):
        for x1 in range(xs.shape[1]):
            u0 = 0.0
            u1 = 0.0
            for k1 in xrange(n):
                for k2 in xrange(n):
                    ps = psis[k1,k2,x0,x1]
                    u0 += u[0,k1,k2] * ps
                    u1 += u[1,k1,k2] * ps
            zs[x0,x1] += np.array([u0, u1])
    return zs

def _calc_psis(d, xs):
    psis = np.empty((d,d) + xs.shape[:2])
    for x0, x1 in product(range(xs.shape[0]), range(xs.shape[1])):
        x = xs[x0,x1] 
        for k1, k2 in product(range(d), repeat=2): 
            psis[k1,k2,x0,x1] = psi(k1, k2, x)
    return psis

def _gen_xs(shape):
    xs = np.empty(shape + (2,))
    for x0, x1 in product(range(shape[0]), range(shape[1])): 
        xs[x0,x1] = np.array([float(x0)/(shape[0]), float(x1)/shape[1]])
    return xs
    

def imagedef(F, I, A=4):
    """
    F: Prototype
    I: Image that will be deformed
    """
    xs = _gen_xs(F.shape)

    delF = np.gradient(F)
    delF[0] /= F.shape[0]
    delF[1] /= F.shape[1]

    allx = list(product(range(xs.shape[0]), range(xs.shape[1])))
     
    # 1.
    rho = 0.1 
    d = scriptN(A)
    u = np.zeros((2, d, d))
    ks1, ks2 = np.mgrid[0:d, 0:d]
    psis = _calc_psis(d, xs)
    m = 0
    a = 0 
    dx = 1.0/(xs.shape[0]*xs.shape[1])
    # Ratio between prior and likelihood is done here. Basically this boils down to the
    # variance of the prior.
    invvar = 1.0 * dx
    stepsize = 0.1
    costs = []
    logpriors = []
    loglikelihoods = []
    for a in range(1, A+1):
        n = scriptN(a)
        allk = list(product(range(n), repeat=2))
        for loop_inner in range(500):
            # 2.

            # Calculate deformed xs
            zs = deform_x(xs, u, scriptN(a), psis)

            # Interpolated F at zs
            Fzs = ag.math.interp2d(zs, F)

            # Interpolate delF at zs 
            delFzs = np.empty((2,) + F.shape) 
            for q in range(2):
                delFzs[q] = ag.math.interp2d(zs, delF[q])

            v = np.zeros((2,)+(d,)*2)
            # 4.
            terms = Fzs - I
            v = np.zeros((2,)+(d,)*2)
            for q in range(2):
                for k1, k2 in allk: 
                    v[q,k1,k2] = (delFzs[q] * terms * psis[k1,k2]).sum()
            
                # This little puppy requires numpy 1.7 - replace when appropriate
                #v[q] = (delFzs[q] * terms * psis2).sum((0,1))
                
            # We didn't multiply by this yet
            v *= dx

            lmbks = invvar * (ks1**2 + ks2**2)**rho

            # Calculate cost, just for sanity check
            if 0:
                logprior = 0.0
                for k1, k2 in allk: 
                    logprior += lmbks[k1,k2] * (u[:,k1,k2]**2).sum()
                logprior /= 2.0

                loglikelihood = 0.0
                for x0, x1 in allx: 
                    loglikelihood += terms[x0,x1]**2 
                loglikelihood *= dx

                if False and loop_outer == 10:
                    plt.quiver(defs[:,:,1], defs[:,:,0])
                    plt.show() 

                # Cost function
                J = logprior + loglikelihood
                #print "Cost:", J, logprior, loglikelihood
                costs.append(J)
                logpriors.append(logprior)
                loglikelihoods.append(loglikelihood)


            # 5. Gradient descent
            u -= stepsize * (lmbks * u + v)

    return u, costs, logpriors, loglikelihoods
     
def deform(I, u):
    """Deform I according to u"""
    im = np.zeros(I.shape)

    xs = _gen_xs(im.shape)

    xs0 = xs[:,:,0]
    xs1 = xs[:,:,1]

    a = u.shape[0]
    psis = _calc_psis(a, xs)
    zs = deform_x(xs, u, scriptN(a), psis)
    im = ag.math.interp2d(zs, I)
    return im

