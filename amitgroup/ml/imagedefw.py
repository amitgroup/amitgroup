from __future__ import division

import numpy as np
import amitgroup as ag
import amitgroup.util
import pywt

__all__ = ['imagedef']

def imagedef(F, I, A=None, stepsize=0.4, coef=1e-5, rho=1.5, tol=1e-7, calc_costs=False):
    """
    Deforms an a prototype image `F` into a data image `I` using a Daubechies wavelet basis and maximum a posteriori. 

    Parameters
    ----------
    F : ndarray
        Prototype image. Array of shape ``(L, L)`` with normalized intensitites. So far, ``L`` has to be a power of two.
    I : ndarray
        Data image that the prototype will try to deform into. Array of shape ``(L, L)``. 
    A : int
        Coefficient depth limit. If None, then naturally bounded by image size.
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
    imdef : DisplacementFieldWavelet
        The deformation in the form of a :class:`DisplacementField`. 
    info : dict
        Dictionary with info:
         * `iterations`: Total number of iterations, ``N``.
         * `iterations_per_level`: Iterations per coarseness level.
         * `logprior`: The value of the log-prior for each iteration. Array of length ``N``.
         * `loglikelihood`: The value of the log-likelihood for each iteration. Array of length ``N``.

    Examples
    --------
    """
    # Speed this up. How?
    """
    Deform an image into a prototype image:

    >>> import amitgroup as ag
    >>> import amitgroup.ml
    >>> import numpy as np
    >>> import matplotlib.pylab as plt

    Load two example faces and perform the deformation:

    >>> F, I = ag.io.load_example('faces2')
    >>> imgdef, info = ag.ml.imagedef(F, I)
    >>> Fdef = imgdef.deform(F)

    Output the results:

    >>> d = dict(interpolation='nearest', cmap=plt.cm.gray)
    >>> plt.figure(figsize=(7,7))
    >>> plt.subplot(221)
    >>> plt.title("Prototype")
    >>> plt.imshow(F, **d)
    >>> plt.subplot(222)
    >>> plt.title("Data image")
    >>> plt.imshow(I, **d) 
    >>> plt.subplot(223)
    >>> plt.title("Deformed")
    >>> plt.imshow(Fdef, **d)
    >>> plt.subplot(224)
    >>> plt.title("Deformation map")
    >>> x, y = imgdef.get_x(F.shape)
    >>> Ux, Uy = imgdef.deform_map(x, y) 
    >>> plt.quiver(y, -x, Uy, -Ux)
    >>> plt.show()
     
    """
    logpriors = []
    loglikelihoods = []

    delF = np.gradient(F)
    # Normalize since the image covers the square around [0, 1].
    delF[0] /= F.shape[0]
    delF[1] /= F.shape[1]
    
    imdef = ag.util.DisplacementFieldWavelet(F.shape, coef=coef, rho=rho)
    
    x0, x1 = imdef.get_x(F.shape)

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
        for loop_inner in xrange(7000): # This number is just a maximum
            num_iterations += 1
            # 2.

            # Calculate deformed xs
            Ux, Uy = imdef.deform_map(x0, x1)
            z0 = x0 + Ux
            z1 = x1 + Uy

            # Interpolate F at zs
            Fzs = ag.util.interp2d(z0, z1, F)

            # Interpolate delF at zs 
            delFzs = np.empty((2,) + F.shape) 
            for q in range(2):
                delFzs[q] = ag.util.interp2d(z0, z1, delF[q], fill_value=0.0)

            # 4.
            terms = Fzs - I
            # Calculate cost, just for sanity check
            loglikelihood = -(terms**2).sum() * dx

            if calc_costs:
                logprior = -imdef.logprior() 
                logpriors.append(logprior)

                loglikelihoods.append(loglikelihood)
        
                if __debug__ and loop_inner%10 == 0:
                    ag.info("cost = {0}".format(-logprior-loglikelihood))
            
            # Check termination
            if loglikelihood - last_loglikelihood <= tol and loop_inner > 100: # Require at least 100
                 break
            last_loglikelihood = loglikelihood

            # 5. Gradient descent
            W = np.empty((2,) + terms.shape)
            for q in range(2):
                W[q] = delFzs[q] * terms
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
