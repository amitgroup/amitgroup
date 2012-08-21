from __future__ import division

import math
import numpy as np
import amitgroup as ag
import amitgroup.util
import pywt

def _powerof2(v):
    # Does not handle 0, but that's not valid for image deformations anyway
    return (v & (v-1)) == 0 

def image_deformation_old(F, I, last_level=None, penalty=1.0, rho=2.0, tol=0.001, \
             wavelet='db2', \
             max_iterations_per_level=1000, start_level=1, stepsize_scale_factor=1.0):
    """
    Deforms an a prototype image `F` into a data image `I` using a Daubechies wavelet basis and maximum a posteriori. 

    Parameters
    ----------
    F : ndarray
        Prototype image. Array of shape ``(W, H)`` with normalized intensitites. Both `W` and `H` must be powers of two.
    I : ndarray
        Data image that the prototype will try to deform into. Array of shape ``(W, H)``. 
    last_level : int
        Coefficient depth limit. If `None`, then naturally bounded by image size. 
        A higher level will allow for finer deformations, but incur a computational overhead.
    penalty : float
        Determines the weight of the prior as opposed to the likelihood. (arbitrarily proportional to the ratio of the inverse variance of the gaussian deformations of the prior and the likelihood). Reduce this value if you want more deformations.
    rho : float
        Determines the penalty of more granular coefficients. Increase to smoothen. Must be strictly positive.
    first_level : int
        First coarse-to-fine coefficient level. Defaults to 1 and generally does not needed to be fiddle with. 
    stepsize_scale_factor : float
        Adjust the inferred stepsize to fine-tune stability and speed of iterations.
    
    Returns
    -------
    imdef : DisplacementFieldWavelet
        The deformation in the form of a :class:`DisplacementField`. 
    info : dict
        Dictionary with info:
         * `iterations`: Total number of iterations, `N`.
         * `iterations_per_level`: Iterations per coarseness level.
         * `logpriors`: The value of the log-prior for each iteration. Array of length `N`.
         * `loglikelihoods`: The value of the log-likelihood for each iteration. Array of length `N`.
         * `costs`: The cost function over time (the negative sum of logprior and loglikelihood)
    """
    # Speed this up. How?
    """
    Examples
    --------
    Deform an image into a prototype image:

    >>> import amitgroup as ag
    >>> import amitgroup.ml
    >>> import numpy as np

    Load two example faces and perform the deformation:

    >>> F, I = ag.io.load_example('faces2')
    >>> imdef, info = ag.stats.image_deformation(F, I)
    >>> Fdef = imdef.deform(F)

    Output the results:

    >>> ag.plot.deformation(F, I, imdef)
     
    """
    assert rho > 0, "Parameter rho must be strictly positive"
    assert len(F.shape) == 2 and len(I.shape) == 2, "Images must be 2D ndarrays"
    assert _powerof2(F.shape[0]) and _powerof2(I.shape[1]), "Image sides must be powers of 2"
    assert F.shape == I.shape, "Images must have the same shape"

    # Shift the penalty value, as to make 1.0 a reasonable value.
    logpriors = []
    loglikelihoods = []


    # Notice that the image is always considered to lie in the range [0, 1] on both axes.
    delF = np.gradient(F, 1/F.shape[0], 1/F.shape[1])
    
    imdef = ag.util.DisplacementFieldWavelet(F.shape, penalty=penalty, rho=rho, wavelet=wavelet, level_capacity=3)
    
    x, y = imdef.meshgrid()

    # 1. 
    dx = 1/np.prod(F.shape)

    last_loglikelihood = -np.inf 
    last_cost = np.inf

    num_iterations = 0
    iterations_per_level = []
    if last_level is None:
        last_level = imdef.levels

    for a in range(start_level, last_level+1):
        ag.info("Running coarse-to-fine level", a)
        for loop_inner in xrange(max_iterations_per_level):
            num_iterations += 1
            # 2. Deform
        
            # Calculate deformed xs
            Ux, Uy = imdef.deform_map(x, y)
            z0 = x + Ux
            z1 = y + Uy

            # Interpolate F at zs
            Fzs = ag.util.interp2d(z0, z1, F)
            #Fzs = imdef.deform(F)

            # Interpolate delF at zs 
            delFzs = np.empty((2,) + F.shape) 
            for q in range(2):
                delFzs[q] = ag.util.interp2d(z0, z1, delF[q], fill_value=0.0)

            # 3. Cost
            terms = Fzs - I

            # Technically we should have a sigma involved here,
            # but we are letting them be absorbed into the ratio
            # between prior and loglikelihood, which we call penalty.
            loglikelihood = -(terms**2).sum() * dx / 2.0
            logprior = imdef.logprior(a)
            logpriors.append(logprior)

            loglikelihoods.append(loglikelihood)

            cost = -logprior - loglikelihood
    
            if __debug__:
                ag.info("cost: {0} (prior: {1}, llh: {2})".format(cost, logprior, loglikelihood))
            
            if math.fabs(cost - last_cost)/last_cost < tol and loop_inner > 5:
                break

            last_cost = cost
            last_loglikelihood = loglikelihood

            # 4. Decide step size (this is done once for each coarse-to-fine level)
            if loop_inner == 0:
                delFzs2 = delFzs[0]**2 + delFzs[1]**2

                # This is the value of the psi, which turns out to be a good bound for this purpose
                M = dx 
                T = a**2 * M * delFzs2.sum() * dx + imdef.sum_of_coefficients(a)

                dt = stepsize_scale_factor/T


            # 5. Gradient descent
            W = np.empty((2,) + terms.shape)
            for q in range(2):
                W[q] = delFzs[q] * terms

            imdef.u -= dt * (imdef.u * imdef.lmbks + imdef.transform(W, a)/4**imdef.levels)

        iterations_per_level.append(num_iterations)
        num_iterations = 0
    
               
    info = {}
    info['iterations_per_level'] = iterations_per_level
    info['iterations'] = sum(iterations_per_level) 
    logpriors = np.asarray(logpriors)
    loglikelihoods = np.asarray(loglikelihoods)
    info['logpriors'] = logpriors
    info['loglikelihoods'] = loglikelihoods
    info['costs'] = -logpriors - loglikelihoods

    return imdef, info
