
from __future__ import division

import numpy as np
import amitgroup as ag
import amitgroup.util
import amitgroup.features
import pywt

twopi = 2.0 * np.pi

# Use ag.ml._deformed_x instead

def _gen_xs(shape):
    dx = 1/shape[0]
    dy = 1/shape[1]
    return np.mgrid[0:1.0-dx:shape[0]*1j, 0:1.0-dy:shape[1]*1j]

def empty_u(tlevels, scriptNs):
    u = []
    for q in range(2):
        u0 = [np.zeros((scriptNs[0],)*2)]
        for a in range(1, tlevels):
            sh = (scriptNs[a],)*2
            u0.append((np.zeros(sh), np.zeros(sh), np.zeros(sh)))
        u.append(u0)
    return u 

def _pywt2array(coef, scriptNs, maxL=1000):
    L = len(scriptNs)#len(coef)
    N = scriptNs[-1]#len(coef[-1][0])
    new_u = np.zeros((L, 3, N, N))
    for i in range(min(maxL, L)):
        if i == 0:
            Nx, Ny = coef[i].shape
            new_u[i,0,:Nx,:Ny] = coef[i]
        else:
            for alpha in range(3):
                Nx, Ny = coef[i][alpha].shape
                new_u[i,alpha,:Nx,:Ny] = coef[i][alpha]
    return new_u

def _array2pywt(coef, scriptNs):
    new_u = []
    for i, N in enumerate(scriptNs): 
        if i == 0:
            new_u.append(coef[i,0,:N,:N])
        else:
            als = []
            for alpha in range(3):
                als.append(coef[i,alpha,:N,:N])
            new_u.append(tuple(als))
    return new_u

def bernoulli_train(data):
    edges = ag.features.bedges(data)
    return edges

def bernoulli_model(F, I, A=None, stepsize=0.1, coef=1e-3, rho=1.5, tol=1e-7, calc_costs=False):
    """
    """
    assert F.shape[1:] == I.shape, "F and I must match"
    assert F.shape[0] == 8, "F must already be the means of edge features"

    logpriors = []
    loglikelihoods = []

    X_ = ag.features.bedges(np.array([I]))
    X = np.rollaxis(X_[0], 2)

    x0, x1 = _gen_xs(I.shape)
    
    if 1:
        all_js = range(8)
        filler = []
    else:
        all_js = [2]
        filler = [None] * 2

    delFjs = [] + filler 
    for j in all_js:
        delF = np.gradient(F[j])
        # Normalize since the image covers the square around [0, 1].
        delF[0] /= F[j].shape[0]
        delF[1] /= F[j].shape[1]
        delFjs.append(delF)
    
    imdef = ag.util.DisplacementFieldWavelet(x0.shape, coef=coef, rho=rho)

    # 1. 
    dx = 1/(x0.shape[0]*x0.shape[1])

    last_loglikelihood = -np.inf 

    num_iterations = 0
    iterations_per_level = []
    # TODO: bad access of imdef.scriptNs
    for a, N in enumerate(imdef.scriptNs): 
        if a == A:
            break
        ag.info("Running coarse-to-fine level", a)
        for loop_inner in xrange(2000): # This number is just a maximum
            num_iterations += 1
            # 2.

            # Calculate deformed xs
            Ux, Uy = imdef.deform_map(x0, x1)
            z0 = x0 + Ux
            z1 = x1 + Uy

            # Interpolate F at zs
            Fjzs = [] + filler 
            for j in all_js: 
                Fzs = ag.math.interp2d(z0, z1, F[j])
                Fjzs.append(Fzs)

            # Interpolate delF at zs 
            delFjzs = [] + filler
            for j in all_js: 
                delFzs = np.empty((2,) + F[j].shape) 
                for q in range(2):
                    #print(delFjs[j][q].shape)
                    #print z0.shape 
                    #print z1.shape
                    delFzs[q] = ag.math.interp2d(z0, z1, delFjs[j][q], fill_value=0.0)
                delFjzs.append(delFzs)

            # 4.
            #terms = Fzs - I
            # Calculate cost, just for sanity check
            #loglikelihood = -(terms**2).sum() * dx
            loglikelihood = 0.0

            if calc_costs:
                logprior = -imdef.logprior() 
                logpriors.append(logprior)

                for j in all_js:
                    loglikelihood = (X[j] * np.log(Fjzs[j]) + 0.1 * (1-X[j]) * np.log(1.0-Fjzs[j])).sum()
                loglikelihoods.append(loglikelihood)
            
            # Check termination
            #print(loglikelihood - last_loglikelihood)
            if loglikelihood - last_loglikelihood <= tol and loop_inner >= 100: # Require at least 100
                 break
            last_loglikelihood = loglikelihood

            W = np.empty((2,) + x0.shape)
            for q in range(2):
                W[q] = 0.0
                for j in all_js: 
                    t1 = X[j] * delFjzs[j][q]/Fjzs[j]
                    t2 = -(1-X[j]) * delFjzs[j][q]/(1-Fjzs[j])
                    W[q] += t1 + t2 

            # 5. Gradient descent
            imdef.reestimate(stepsize, W, a+1)

        iterations_per_level.append(num_iterations)
        num_iterations = 0
    
               
    info = {}
    info['iterations_per_level'] = iterations_per_level
    info['iterations'] = sum(iterations_per_level) 
    if calc_costs:
        info['logpriors'] = np.array(logpriors)
        info['loglikelihoods'] = np.array(loglikelihoods)

    return imdef, info 
