
from __future__ import division

import numpy as np
import amitgroup as ag
import pywt

__all__ = ['imagedef']

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

def imagedef(F, I, A=None, stepsize=0.1, coef=1e-3, rho=1.5, tol=1e-7, calc_costs=False):
    """
    Deforms an image ``I`` into a prototype image ``F`` using a Daubechies wavelet basis and maximum a posteriori. 

    Parameters
    ----------
    F : ndarray
        Prototype image. Array of shape ``(L, L)`` with normalized intensitites. So far, ``L`` has to be a power of two.
    I : ndarray
        Image that will be deformed. Array of shape ``(L, L)``. 
    A : int
        Coefficient depth limit. If None, unlimited.
    stepsize : float
        Gradient descent step size.
    coef : float
        Determines the weight of the prior (proportional to the inverse variance of the gaussian deformations).
    rho : float
        Determines the penalty of more granular coefficients. Increase to smoothen.
    calc_costs : bool
        If True, then ``info`` will contain `logprior` and `loglikelihood`. The cost function `J` is the negative sum of these two. 
    
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

    Load two example faces and perform the deformation:

    >>> im1, im2 = ag.io.load_example('faces2')
    >>> imgdef, info = ag.ml.imagedef(im1, im2)
    >>> im3 = imgdef.deform(im1)

    Output the results:

    >>> d = dict(interpolation='nearest', cmap=plt.cm.gray)
    >>> plt.figure(figsize=(7,7))
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
    >>> x, y = imgdef.get_x(im1.shape)
    >>> Ux, Uy = imgdef.deform_map(x, y) 
    >>> plt.quiver(y, -x, Uy, -Ux)
    >>> plt.show()
     
    """
    logpriors = []
    loglikelihoods = []

    x0, x1 = _gen_xs(F.shape)

    delF = np.gradient(F)
    # Normalize since the image covers the square around [0, 1].
    delF[0] /= F.shape[0]
    delF[1] /= F.shape[1]
    
    imdef = ag.IDWavelet(x0.shape, coef=coef, rho=rho)

    # 1. 
    dx = 1/(x0.shape[0]*x0.shape[1])

    last_loglikelihood = -np.inf 

    num_iterations = 0
    iterations_per_level = []
    # TODO: bad access of imdef.scriptNs
    for a, N in enumerate(imdef.scriptNs): 
        if a == A:
            break
        # TODO: Add a ag.VERBOSE, or better yet, a ag.verbose(...)
        ag.info("Running coarse-to-fine level", a)
        for loop_inner in xrange(4000): # This number is just a maximum
            num_iterations += 1
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
            loglikelihood = -(terms**2).sum() * dx

            if calc_costs:
                logprior = -imdef.logprior() 
                logpriors.append(logprior)

                loglikelihoods.append(loglikelihood)
            
            # Check termination
            if loglikelihood - last_loglikelihood <= tol and loop_inner > 100: # Require at least 100
                 break
            last_loglikelihood = loglikelihood

            # 5. Gradient descent
            imdef.reestimate(stepsize, delFzs, Fzs, I, a+1)

        iterations_per_level.append(num_iterations)
        num_iterations = 0
    
               
    info = {}
    info['iterations_per_level'] = iterations_per_level
    info['iterations'] = sum(iterations_per_level) 
    if calc_costs:
        info['logpriors'] = np.array(logpriors)
        info['loglikelihoods'] = np.array(loglikelihoods)

    return imdef, info 
