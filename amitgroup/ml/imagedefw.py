
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

def psi(k1, k2, x):
    return twopi * cos(twopi * (k1 * x[0] + k2 * x[1]))

def scriptN(a):
    return a 


def deform_map(xs, u):
    #print u[0][0].shape
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
def deform_x(xs, u):
    zs = np.copy(xs) 
    #print(u)

    zs += deform_map(xs, u) 

    #for x0 in range(xs.shape[0]):
    #    for x1 in range(xs.shape[1]):
    #        u0 = 0.0
    #        u1 = 0.0
    #        for s in xrange(S):
    #            L = 4**s
    #            for l1 in xrange(L):
    #                for l2 in xrange(L):
    #                    #ps = psis[k1,k2,x0,x1]
    #                    u0 += u[0,s,l1,l2] * psi(
    #                    u1 += u[1,s,l1,l2] * 
    #        zs[x0,x1] += np.array([u0, u1])
    return zs

def fix2(x):
    L = len(x)
    x[0] /= 2.0**((L-1)/1.0)
    for i in range(1, L):
        x[i] = list(x[i])
        for alpha in range(3):
            x[i][alpha] /= 2.0**((L-i)/1.0)
        x[i] = tuple(x[i])
    return x

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

def empty_u(levels):
    u = []
    for q in range(2):
        u0 = [np.zeros((scriptNs[0],)*2)]
        for a in range(1, levels+1):
            sh = (scriptNs[a],)*2
            u0.append((np.zeros(sh), np.zeros(sh), np.zeros(sh)))
        u.append(u0)
    return u 
    
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
    levels = len(pywt.wavedec(range(32), wl_name)) - 1
    scriptNs = map(len, pywt.wavedec(range(32), wl_name, level=levels)) + [0]

    u = empty_u(levels)

    dx = 1.0/(xs.shape[0]*xs.shape[1])
    # Ratio between prior and likelihood is done here. Basically this boils down to the
    # variance of the prior.
    invvar = dx
    stepsize = 0.1
    costs = []
    logpriors = []
    loglikelihoods = []

    ll_boost = 1.0

    lmbks = np.zeros(levels+1)
    for i in range(levels+1):
        lmbks[i] = 1.0 * invvar * 2.0**(rho * i) 

    for a in range(0, 20): 
        if scriptNs[a] == 0 :#or a == 3:
            break
        print "-------- a = {0} ---------".format(a)
        #n = scriptN(S)
        #allk = list(product(range(n), repeat=2))
        for loop_inner in xrange(2000): 
            # 2.

            # Calculate deformed xs
            zs = deform_x(xs, u)

            # Interpolated F at zs
            Fzs = ag.math.interp2d(zs, F)

            # Interpolate delF at zs 
            delFzs = np.empty((2,) + F.shape) 
            for q in range(2):
                delFzs[q] = ag.math.interp2d(zs, delF[q], fill_value=0.0)

            # 4.
            terms = Fzs - I
            #v = np.zeros((2,)+(d,)*2)
            #for q in range(2):
            #    for s in range(A):
            #        L = 2**s
            #        for l1 in range(L):
            #            for l2 in range(L):
            #                v[q,k1,k2] = (delFzs[q] * terms * psis[k1,k2]).sum()
            #    pass  
            
                # This little puppy requires numpy 1.7 - replace when appropriate
                #v[q] = (delFzs[q] * terms * psis2).sum((0,1))
                
            # We didn't multiply by this yet
            #v *= dx

            #lmbks = invvar * (ks1**2 + ks2**2)**rho

            # Calculate cost, just for sanity check
            if 1:
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
            new_u = deepcopy(u)
        
            #lmbks = invvar * (ks1**2 + ks2**2)**rho
            
            for q in range(2):
                f = delFzs[q] * terms
                #level = 1
                #while True:
                #    coef = pywt.wavedec2(f, wl_name, level=level)
                #    if len(coef[0]) == 1:
                #        break
                #    level += 1

                coef = pywt.wavedec2(f, wl_name, level=levels)
                #coef = fix2(coef)
                    
                #lmbk = lmbks[0]
                #vqk = 0.0
                #for x0 in range(xs.shape[0]):
                #    for x1 in range(xs.shape[1]):
                #        p = 1.0
                #        vqk += delFzs[q,x0,x1] * (Fzs[x0,x1] - I[x0,x1]) * p * dx

                #print 'HERE:' 
                #print vqk
                #print coef[0][0,0]
                #print 'QQ', coef[0][0,0]/vqk
                #print '......'

                # boost?
                #vqk *= ll_boost 
            
                #print vqk
                #new_u[q][0] -= stepsize * (lmbk * u[q][0] + vqk)

                #print coef[:1]

                #import sys; sys.exit(0); 

                if 0:
                    import matplotlib.pylab as plt
                    plt.figure()
                    plt.imshow(coef[0], interpolation='nearest', cmap=plt.cm.gray)
                    plt.colorbar()
                    plt.show()

                if 0:
                    import matplotlib.pylab as plt
                    plt.imshow(f, interpolation='nearest', cmap=plt.cm.gray)
                    plt.colorbar()
                    plt.show()
                    import sys
                    sys.exit(0)
    
                for i in range(a+1):
                    lmbk = lmbks[i]
                    L = scriptNs[i]
                    for l1 in range(L):
                        for l2 in range(L):
                            if i == 0:
                                vqk = coef[i][l1,l2] * ll_boost 
                                new_u[q][i][l1,l2] -= stepsize * (lmbk * u[q][i][l1,l2] + vqk)
                            elif i != 0:
                                for alpha in range(3):
                                    vqk = coef[i][alpha][l1,l2] * ll_boost 
                                    new_u[q][i][alpha][l1,l2] -= stepsize * (lmbk * u[q][i][alpha][l1,l2] + vqk)
            
            u = new_u

            #print "{0:3f} {1:3f} {2:3f}".format(u[0][0][2,0], u[0][0][1,0], u[0][0][0,0])
            #print u[0][0][2,0]
            #print u[0][0], u[1][0]
            #print S
            #print u[0][0], u[0][1:S+1]
            #print u[1][0], u[1][1:S+1]

    #print u[0][0]
    import pickle
    pickle.dump(u, open('u.p', 'wb'))  
    return u, costs, logpriors, loglikelihoods
     
def deform(I, u):
    """Deform I according to u"""
    im = np.zeros(I.shape)

    xs = _gen_xs(im.shape)

    xs0 = xs[:,:,0]
    xs1 = xs[:,:,1]

    #psis = _calc_psis(a, xs)
    zs = deform_x(xs, u)
    im = ag.math.interp2d(zs, I)
    return im

