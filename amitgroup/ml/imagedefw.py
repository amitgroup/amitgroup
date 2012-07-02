
import numpy as np
import amitgroup as ag
from amitgroup.ml.aux import deform_x as deform_x_old
from itertools import product
from math import cos
import pywt
from copy import deepcopy


__all__ = ['imagedef', 'deform', 'deform_x', 'deform_map']

twopi = 2.0 * np.pi

def psi(k1, k2, x):
    return twopi * cos(twopi * (k1 * x[0] + k2 * x[1]))

def scriptN(a):
    return a 


def deform_map(xs, u):
    defx0 = pywt.waverec2(u[0], 'haar')
    defx1 = pywt.waverec2(u[1], 'haar')

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
    rho = 1.0 
    d = scriptN(A)
    A = 2 
    #u = np.zeros((2, d, d))
    u = []
    for q in range(2):
        u0 = [np.zeros((1,1))]
        for s in range(0, A):
            sh = (2**s, 2**s)
            u0.append((np.zeros(sh), np.zeros(sh), np.zeros(sh)))
        u.append(u0)
    
    #print u
    ks1, ks2 = np.mgrid[0:d, 0:d]
    psis = _calc_psis(d, xs)
    m = 0
    S = 0 
    dx = 1.0/(xs.shape[0]*xs.shape[1])
    # Ratio between prior and likelihood is done here. Basically this boils down to the
    # variance of the prior.
    invvar = 1.0 * dx
    stepsize = 0.1
    costs = []
    logpriors = []
    loglikelihoods = []
    for S in range(0, A):
        print "-------- S = {0} ---------".format(S)
        #n = scriptN(S)
        #allk = list(product(range(n), repeat=2))
        for loop_inner in range({0:100, 1:500}[S]):
            # 2.

            # Calculate deformed xs
            zs = deform_x(xs, u)

            # Interpolated F at zs
            Fzs = ag.math.interp2d(zs, F)

            # Interpolate delF at zs 
            delFzs = np.empty((2,) + F.shape) 
            for q in range(2):
                delFzs[q] = ag.math.interp2d(zs, delF[q])

            v = np.zeros((2,)+(d,)*2)
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
            new_u = deepcopy(u)
        
            #lmbks = invvar * (ks1**2 + ks2**2)**rho
            
            for q in range(2):
                f = delFzs[q] * terms
                level = 1
                while True:
                    coef = pywt.wavedec2(f, 'db1', level=level)
                    if len(coef[0]) == 1:
                        break
                    level += 1

                coef = fix2(coef)
                    
                lmbk = invvar * 2.0**(rho * 0)
                vqk = 0.0
                for x0 in range(xs.shape[0]):
                    for x1 in range(xs.shape[1]):
                        p = 1.0
                        vqk += delFzs[q,x0,x1] * (Fzs[x0,x1] - I[x0,x1]) * p * dx

                print 'HERE:' 
                print vqk
                print coef[0][0,0]
                print 'QQ', coef[0][0,0]/vqk
                print '......'

                # boost?
                vqk *= 100.0
            
                #print vqk
                new_u[q][0] -= stepsize * (lmbk * u[q][0] + vqk)

                def psi(t):
                    if 0 <= t < 0.5:
                        return 1
                    elif 0.5 <= t < 1:
                        return -1
                    else:
                        return 0
    
                if 0:
                    import matplotlib.pylab as plt
                    plt.imshow(f, interpolation='nearest', cmap=plt.cm.gray)
                    plt.colorbar()
                    plt.show()
                    import sys
                    sys.exit(0)
    
                for s in range(1, S+1):
                    for alpha in range(3):
                        L = 2**(s-1)
                        lmbk =  2.0**(rho * s)
                        for l1 in range(L):
                            for l2 in range(L):
                                #vqk = 0.0
                                #for x0 in range(xs.shape[0]):
                                #    for x1 in range(xs.shape[1]):
                                #        x = xs[x0,x1]
                                #        p = psi(x[0]) * psi(x[1])
                                #        vqk += delFzs[q,x0,x1] * terms[x0,x1] * p * dx
                                #v[q,k1,k2] = (delFzs[q] * terms * psis[k1,k2]).sum()
                                #pass#u[q][
                                vqk = coef[s][alpha][l1,l2]
                                if (q, alpha, l1, l2) == (1, 1, 0, 0):
                                    print 'vqk', vqk
                                #vqk *= 100.0

                                #vqk *= 1000.0
                                new_u[q][s][alpha][l1,l2] -= stepsize * (lmbk * u[q][s][alpha][l1,l2] + vqk)
            
            #u -= stepsize * (lmbks * u + v)
            u = new_u
            #print u[0][0], u[1][0]
            print S
            print u[0][0], u[0][1:S+1]
            print u[1][0], u[1][1:S+1]

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

