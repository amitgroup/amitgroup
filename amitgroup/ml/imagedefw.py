
import numpy as np
import amitgroup as ag
from itertools import product
from math import cos
import pywt
from copy import deepcopy


__all__ = ['imagedef']

twopi = 2.0 * np.pi

# Use ag.ml._deformed_x instead

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

    Examples
    --------
    Deform an image into a prototype image:

    >>> import amitgroup as ag
    >>> import numpy as np
    >>> import matplotlib.pylab as plt
    >>> ims = ag.io.load_example('faces2')
    >>> im1 = ims[0]
    >>> im2 = ims[1]
    >>> im1 = im1[::-1,:]
    >>> im2 = im2[::-1,:]
    >>> imgdef, info = ag.ml.imagedef(im1, im2, rho=3.0)
    >>> im3 = imgdef.deform(im1)
    >>> x, y = imgdef.get_x(im1.shape)
    >>> Ux, Uy = imgdef.deform_map(x, y) 
    >>> d = dict(origin='lower', interpolation='nearest', cmap=plt.cm.gray)
    >>> plt.subplot(221)
    >>> plt.title("Prototype")
    >>> plt.imshow(im1, **d)
    >>> plt.subplot(222)
    >>> plt.title("Original")
    >>> plt.imshow(im2, **d) 
    >>> plt.subplot(223)
    >>> plt.title("Deformed")
    >>> plt.imshow(im3, **d)
    >>> plt.subplot(224)
    >>> plt.title("Deformation map")
    >>> plt.quiver(y, x, Uy, Ux)
    >>> plt.show()
     
    """
    logpriors = []
    loglikelihoods = []

    x0, x1 = _gen_xs(F.shape)

    delF = np.gradient(F)
    delF[0] /= F.shape[0]
    delF[1] /= F.shape[1]
    
    imdef = ag.IDWavelet(x0.shape, rho=rho)

    # 1. 

    dx = 1.0/(x0.shape[0]*x0.shape[1])
    # Ratio between prior and likelihood is done here. Basically this boils down to the
    # variance of the prior.
    invvar = dx
    stepsize = 0.1

    total_iterations = 0
    # TODO: bad access of imdef.scriptNs
    for a, N in enumerate(imdef.scriptNs): 
        if a == A:
            break
        print "-------- a = {0} ---------".format(a)
        for loop_inner in xrange(2000): 
            total_iterations += 1
            # 2.

            # Calculate deformed xs
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
                logprior = imdef.logprior() 
                logpriors.append(logprior)

                loglikelihood = (terms**2).sum() * dx
                loglikelihoods.append(loglikelihood)

            # 5. Gradient descent
            imdef.reestimate(stepsize, delFzs, Fzs, I, a+1)
    
               
    info = {}
    info['iterations'] = total_iterations
    if calc_costs:
        info['logpriors'] = np.array(logpriors)
        info['loglikelihoods'] = np.array(loglikelihoods)

    return imdef, info 
