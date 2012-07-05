
import numpy as np
import amitgroup as ag
from itertools import product
from math import cos
import pywt
from copy import deepcopy


__all__ = ['imagedef', 'deform', '_deformed_x', 'deform_map']

wl_name = 'db2'

twopi = 2.0 * np.pi

def deform_map(x, y, u, scriptNs):
    """
    Creates a deformation array according the coefficients from imagedef. 

    Parameters
    ----------
    x, y : ndarray
        Arrays of `x` and `y` values. Generate these by ``numpy.mgrid``. Array of shape ``(L, L)``.
    u : ndarray
        Array of coefficients. Returns by amitgroup.ml.imagedef.
    scriptNs:
        TODO: Should be described inside ``u``, and thus not be needed.
    
    Returns
    -------
    Ux : ndarray
        Deformation along the `x` axis. Array of shape ``(L, L)``. 
    Uy : ndarray
        Same as above, along `y` axis. 
    """
    defx0 = pywt.waverec2(_array2pywt(u[0], scriptNs), wl_name) 
    defx1 = pywt.waverec2(_array2pywt(u[1], scriptNs), wl_name)

    # Interpolated defx at xs 
    Ux = ag.math.interp2d(x, y, defx0, dx=np.array([1.0/(defx0.shape[0]-1), 1.0/(defx0.shape[1]-1)]))
    Uy = ag.math.interp2d(x, y, defx1, dx=np.array([1.0/(defx1.shape[0]-1), 1.0/(defx1.shape[1]-1)]))
    return Ux, Uy 

# Use ag.ml._deformed_x instead
def _deformed_x(x0, x1, u, scriptNs):
    Ux0, Ux1 = deform_map(x0, x1, u, scriptNs)
    return x0+Ux0, x1+Ux1

def _gen_xs(shape):
    return np.mgrid[0:1.0:shape[0]*1j, 0:1.0:shape[1]*1j]

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

def imagedef(F, I, A=None, rho=1.5, calc_costs=False):
    """
    Deforms an image ``I`` into a prototype image ``F`` using a Daubechies wavelet basis and minimizing the posterior distribution. 

    Parameters
    ----------
    F : ndarray
        Prototype image. Array of shape ``(L, L)`` with normalized intensitites. So far, ``L`` has to be a power of two.
    I : ndarray
        Image that will be deformed. Array of shape ``(L, L)``. 
    A : int
        Coefficient depth limit. If None, unlimited.
    rho : float
        Determines the penalty of more granular coefficients. Increase to smoothen.
    calc_costs : bool
        If True, then ``info`` will contain `logprior` and `loglikelihood`. The cost function `J` is simply the sum of these two. 
    
    Returns
    -------
    u : ndarray
        The deformation coefficients of the Daubechies (D4) wavelet.
    info : dict
        Dictionary with info:
        - `iteratons`: Total number of iterations, ``N``.
        - `logprior`: The value of the log-prior for each iteration. Array of length ``N``.
        - `loglikelihood`: The value of the log-likelihood for each iteration. Array of length ``N``.

    """
    logpriors = []
    loglikelihoods = []

    x0, x1 = _gen_xs(F.shape)

    delF = np.gradient(F)
    delF[0] /= F.shape[0]
    delF[1] /= F.shape[1]
    
    imdef = ag.IDWavelet()

    # 1. 

    # Arrange scriptNs
    tlevels = len(pywt.wavedec(range(32), wl_name))
    levels = tlevels - 1
    scriptNs = map(len, pywt.wavedec(range(32), wl_name, level=levels))
    biggest = scriptNs[-1]

    imdef.scriptNs = scriptNs
    
    # Shape of the coefficients array. This array will be largely under-utilized, since
    # not all levels will have biggest*biggest coefficients. This could be optimized to a
    # a single list, but at a small scale, this does not require too much memory.
    ushape = (2, tlevels, 3, biggest, biggest)

    u = np.zeros(ushape)
    imdef.u = u

    dx = 1.0/(x0.shape[0]*x0.shape[1])
    # Ratio between prior and likelihood is done here. Basically this boils down to the
    # variance of the prior.
    invvar = dx
    stepsize = 0.1

    lmbks = np.zeros(ushape)
    for i in range(tlevels):
        lmbks[:,i,:,:,:] = invvar * 2.0**(rho * i) 

    total_iterations = 0
    for a, N in enumerate(scriptNs): 
        if a == A:
            break
        print "-------- a = {0} ---------".format(a)
        for loop_inner in xrange(20): 
            total_iterations += 1
            # 2.

            # Calculate deformed xs
            #z0, z1 = _deformed_x(x0, x1, u, scriptNs)
            Ux, Uy = imdef.deform_map(x0, x1)
            z0 = x0 + Ux
            z1 = x1 + Uy
            

            # Interpolate F at zs
            Fzs = ag.math.interp2d(z0, z1, F)

            # Interpolate delF at zs 
            delFzs = np.empty((2,) + F.shape) 
            for q in range(2):
                delFzs[q] = ag.math.interp2d(z0, z1, delF[q], fill_value=0.0)

            # 4.
            terms = Fzs - I
            # Calculate cost, just for sanity check
            if calc_costs:
                logprior = (lmbks * (u**2)).sum() / 2.0
                logpriors.append(logprior)

                loglikelihood = (terms**2).sum() * dx
                loglikelihoods.append(loglikelihood)

            # 5. Gradient descent
            vqks = np.array([
                _pywt2array(pywt.wavedec2(delFzs[q] * terms, wl_name, level=levels), scriptNs, a+1) for q in range(2)
            ])

            #imdef.reestimate(stepsize, 
            #u = u.reestimate(stepsize, lmbks, u)
            u -= stepsize * (lmbks * u + vqks)
    
               
    info = {}
    info['iterations'] = total_iterations
    if calc_costs:
        info['logpriors'] = np.array(logpriors)
        info['loglikelihoods'] = np.array(loglikelihoods)

    return u, info 
     
def deform(I, u, scriptNs):
    """
    Deform I according to Daubechies coefficients u.
    """
    im = np.zeros(I.shape)

    x0, x1 = _gen_xs(im.shape)

    z0, z1 = _deformed_x(x0, x1, u, scriptNs)
    im = ag.math.interp2d(z0, z1, I)
    return im
