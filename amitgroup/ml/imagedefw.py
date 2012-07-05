
import numpy as np
import amitgroup as ag
from amitgroup.ml.aux import deform_x as deform_x_old
from itertools import product
from math import cos
import pywt
from copy import deepcopy


__all__ = ['imagedef', 'deform', 'deform_x', 'deform_map']

wl_name = 'db2'

twopi = 2.0 * np.pi

def deform_map__new(xs, u, scriptNs):
    #print u[0][0].shape
    
    defx0 = pywt.waverec2(denumpyfy_u(u[0], scriptNs), wl_name) 
    defx1 = pywt.waverec2(denumpyfy_u(u[1], scriptNs), wl_name)
    #defx0 = pywt.waverec2(u[0], wl_name) 
    #defx1 = pywt.waverec2(u[1], wl_name)

    #print defx0
    #print defx1

    # Interpolated defx at xs 
    Ux0 = ag.math.interp2d(xs, defx0, dx=np.array([1.0/(defx0.shape[0]-1), 1.0/(defx0.shape[1]-1)]))
    Ux1 = ag.math.interp2d(xs, defx1, dx=np.array([1.0/(defx1.shape[0]-1), 1.0/(defx1.shape[1]-1)]))
    #Ux0 = ag.math.interp2d(xs, defx0)
    #Ux1 = ag.math.interp2d(xs, defx1)
    defx = np.zeros(xs.shape)
    #print xs.shape
    #print Ux.shape
    #zs = xs + Ux
    # TODO: Do this with numpy operations
    for x0 in range(xs.shape[0]):
        for x1 in range(xs.shape[1]):
            defx[x0,x1] = np.array([Ux0[x0,x1], Ux1[x0,x1]])
    return defx
def deform_map(xs, u, scriptNs):
    #print u[0][0].shape
    
    #defx0 = pywt.waverec2(denumpyfy_u(u[0], scriptNs), wl_name) 
    #defx1 = pywt.waverec2(denumpyfy_u(u[1], scriptNs), wl_name)
    defx0 = pywt.waverec2(u[0], wl_name) 
    defx1 = pywt.waverec2(u[1], wl_name)

    #print defx0
    #print defx1

    # Interpolated defx at xs 
    Ux0 = ag.math.interp2d(xs, defx0, dx=np.array([1.0/(defx0.shape[0]-1), 1.0/(defx0.shape[1]-1)]))
    Ux1 = ag.math.interp2d(xs, defx1, dx=np.array([1.0/(defx1.shape[0]-1), 1.0/(defx1.shape[1]-1)]))
    #Ux0 = ag.math.interp2d(xs, defx0)
    #Ux1 = ag.math.interp2d(xs, defx1)
    defx = np.zeros(xs.shape)
    #print xs.shape
    #print Ux.shape
    #zs = xs + Ux
    # TODO: Do this with numpy operations
    for x0 in range(xs.shape[0]):
        for x1 in range(xs.shape[1]):
            defx[x0,x1] = np.array([Ux0[x0,x1], Ux1[x0,x1]])
    return defx

# Use ag.ml.deform_x instead
def deform_x__new(xs, u, scriptNs):
    return xs + deform_map__new(xs, u, scriptNs)
def deform_x(xs, u, scriptNs):
    return xs + deform_map(xs, u, scriptNs)

def _gen_xs(shape):
    xs = np.empty(shape + (2,))
    for x0, x1 in product(range(shape[0]), range(shape[1])): 
        xs[x0,x1] = np.array([float(x0)/(shape[0]), float(x1)/shape[1]])
    return xs

def empty_u(tlevels, scriptNs):
    u = []
    for q in range(2):
        u0 = [np.zeros((scriptNs[0],)*2)]
        for a in range(1, tlevels):
            sh = (scriptNs[a],)*2
            u0.append((np.zeros(sh), np.zeros(sh), np.zeros(sh)))
        u.append(u0)
    return u 

def numpyfy_u(coef, scriptNs):
    L = len(scriptNs)#len(coef)
    N = scriptNs[-1]#len(coef[-1][0])
    new_u = np.zeros((L, 3, N, N))
    for i in range(L):
        for l1 in range(scriptNs[i]):
            for l1 in range(scriptNs[i]):
                if i == 0:
                    Nx, Ny = coef[i].shape
                    #print Nx, Ny
                    #print new_u[i,0].shape #,0:Nx,0:Ny].shape
                    #print coef[i].shape
                    new_u[i,0,:Nx,:Ny] = coef[i]
                else:
                    for alpha in range(3):
                        Nx, Ny = coef[i][alpha].shape
                        new_u[i,alpha,:Nx,:Ny] = coef[i][alpha]
    return new_u

def denumpyfy_u(coef, scriptNs):
    new_u = []
    for i, N in enumerate(scriptNs[:-1]): 
        if i == 0:
            new_u.append(coef[i,0,:N,:N])
        else:
            als = []
            for alpha in range(3):
                als.append(coef[i,alpha,:N,:N])
            new_u.append(tuple(als))
    return new_u

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
    rho = 1.5 

    # Arrange scriptNs
    tlevels = len(pywt.wavedec(range(32), wl_name))
    levels = tlevels - 1
    #print levels
    scriptNs = map(len, pywt.wavedec(range(32), wl_name, level=levels))
    #scriptNs = map(len, pywt.wavedec(range(32), wl_name, level=levels)) + [0]
    biggest = scriptNs[-1]
    #biggest = scriptNs[-2]

    ushape = (2, tlevels, 3, biggest, biggest)

    u__new = np.zeros(ushape)
    u = empty_u(tlevels, scriptNs)

    dx = 1.0/(xs.shape[0]*xs.shape[1])
    # Ratio between prior and likelihood is done here. Basically this boils down to the
    # variance of the prior.
    invvar = dx
    stepsize = 0.1
    costs = []
    logpriors = []
    loglikelihoods = []

    lmbks = np.zeros(tlevels)
    lmbks__new = np.zeros(ushape)
    for i in range(tlevels):
        lmbks[i] = 1.0 * invvar * 2.0**(rho * i)
        lmbks__new[:,i,:,:,:] = 1.0 * invvar * 2.0**(rho * i) 

    print scriptNs
    #import sys; sys.exit(0)
    for a, N in enumerate(scriptNs): 
        if N == 0:
            break
        print "-------- a = {0} ---------".format(a)
        #n = scriptN(S)
        #allk = list(product(range(n), repeat=2))
        for loop_inner in xrange(50): 
            # 2.

            # Calculate deformed xs
            zs = deform_x(xs, u, scriptNs)

            # Interpolated F at zs
            Fzs = ag.math.interp2d(zs, F)

            # Interpolate delF at zs 
            delFzs = np.empty((2,) + F.shape) 
            for q in range(2):
                delFzs[q] = ag.math.interp2d(zs, delF[q], fill_value=0.0)

            # 4.
            terms = Fzs - I
            # Calculate cost, just for sanity check
            if 0:
                logprior = 0.0
                if 1:
                    for q in range(2):
                        for i in range(a+1):
                            L = scriptNs[i] 
                            for l1 in range(L):
                                for l2 in range(L):
                                    if i == 0:
                                        logprior += lmbks[i] * (u[q][i][l1,l2]**2)
                                    else:
                                        for alpha in range(3):
                                            logprior += lmbks[i] * (u[q][i][alpha][l1,l2]**2)
                                        
                logprior /= 2.0

                loglikelihood = (terms**2).sum() * dx

                #if False and loop_outer == 10:
                #    plt.quiver(defs[:,:,1], defs[:,:,0])
                #    plt.show() 

                # Cost function
                J = logprior + loglikelihood
                #print "Cost:", J, logprior, loglikelihood
                costs.append(J)
                logpriors.append(logprior)
                loglikelihoods.append(loglikelihood)

            # 5. Gradient descent
        
            #lmbks = invvar * (ks1**2 + ks2**2)**rho

            #vqks = np.zeros(ushape)
            
            if 1:
                new_u = deepcopy(u)
                for q in range(2):
                    f = delFzs[q] * terms
                    coef = pywt.wavedec2(f, wl_name, level=levels)
                        
                    for i in range(a+1):
                        lmbk = lmbks[i]
                        L = scriptNs[i]
                        for l1 in range(L):
                            for l2 in range(L):
                                if i == 0:
                                    #vqks[q,i,0,l1,l2] = coef[i][l1,l2]
                                    vqk = coef[i][l1,l2]
                                    #print new_u, q, i, l1, l2
                                    #print new_u[q][i][l1,l2]
                                    new_u[q][i][l1,l2] -= stepsize * (lmbk * u[q][i][l1,l2] + vqk)
                                elif i != 0:
                                    for alpha in range(3):
                                        #vqks[q,i,alpha,l1,l2] = coef[i][alpha][l1,l2]
                                        vqk = coef[i][alpha][l1,l2]
                                        #print q, i, alpha, l1, l2
                                        #print len(new_u[q])
                                        new_u[q][i][alpha][l1,l2] -= stepsize * (lmbk * u[q][i][alpha][l1,l2] + vqk)
                u = new_u

            if 1: 
                vqks = np.array([
                    numpyfy_u(pywt.wavedec2(delFzs[q] * terms, wl_name, level=levels), scriptNs) for q in range(2)
                ])

                #print u.shape, lmbks.shape, vqks.shape
                u__new -= stepsize * (lmbks__new * u__new + vqks)
               
        

    #print u[0][0]
    #import pickle
    #pickle.dump(u, open('u.p', 'wb'))  
    return u, costs, logpriors, loglikelihoods#, u__new
     
     
def deform__new(I, u, scriptNs):
    """Deform I according to u"""
    im = np.zeros(I.shape)

    xs = _gen_xs(im.shape)

    xs0 = xs[:,:,0]
    xs1 = xs[:,:,1]

    zs = deform_x__new(xs, u, scriptNs)
    im = ag.math.interp2d(zs, I)
    return im

def deform(I, u, scriptNs):
    """Deform I according to u"""
    im = np.zeros(I.shape)

    xs = _gen_xs(im.shape)

    xs0 = xs[:,:,0]
    xs1 = xs[:,:,1]

    zs = deform_x(xs, u, scriptNs)
    im = ag.math.interp2d(zs, I)
    return im

