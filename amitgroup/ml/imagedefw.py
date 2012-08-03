from __future__ import division

import math
import numpy as np
import amitgroup as ag
import amitgroup.util
import pywt

def _powerof2(v):
    # Does not handle 0, but that's not valid for image deformations anyway
    return (v & (v-1)) == 0 

def imagedef(F, I, last_level=None, penalty=1.0, rho=2.0, tol=0.001, \
             calc_costs=False, wavelet='db2', \
             max_iterations_per_level=1000, start_level=1, stepsize_scale_factor=1.0):
    """
    Deforms an a prototype image `F` into a data image `I` using a Daubechies wavelet basis and maximum a posteriori. 

    Parameters
    ----------
    F : ndarray
        Prototype image. Array of shape ``(L, L)`` with normalized intensitites. So far, ``L`` has to be a power of two.
    I : ndarray
        Data image that the prototype will try to deform into. Array of shape ``(L, L)``. 
    last_level : int
        Coefficient depth limit. If `None`, then naturally bounded by image size. 
        A higher level will allow for finer deformations, but incur a computational overhead.
    penalty : float
        Determines the weight of the prior as opposed to the likelihood. (arbitrarily proportional to the ratio of the inverse variance of the gaussian deformations of the prior and the likelihood). Reduce this value if you want more deformations.
    rho : float
        Determines the penalty of more granular coefficients. Increase to smoothen. Must be strictly positive.
    calc_costs : bool
        If True, then ``info`` will contain `logprior` and `loglikelihood`. The cost function `J` is the negative sum of these two. 
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
         * `iterations`: Total number of iterations, ``N``.
         * `iterations_per_level`: Iterations per coarseness level.
         * `logpriors`: The value of the log-prior for each iteration. Array of length ``N``.
         * `loglikelihoods`: The value of the log-likelihood for each iteration. Array of length ``N``.
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
    >>> imdef, info = ag.ml.imagedef(F, I)
    >>> Fdef = imdef.deform(F)

    Output the results:

    >>> ag.plot.deformation(F, I, imdef)
     
    """
    """Old stuff:
    stepsize : float or None
        Gradient descent step size. If `None`, then the stepsize will be inferred by something along the lines of Newton's method.
    """
    assert rho > 0, "Parameter rho must be strictly positive"
    assert len(F.shape) == 2 and len(I.shape) == 2, "Images must be 2D ndarrays"
    assert _powerof2(F.shape[0]) and _powerof2(I.shape[1]), "Image sides must be powers of 2"
    assert F.shape == I.shape, "Images must have the same shape"

    # This should work, but is very wonky. Needs more testing.
    assert F.shape[0] == F.shape[1], "Sides must be the same length... For now. Will fix."

    # Shift the penalty value, as to make 1.0 a reasonable value.
    penalty_adjusted = penalty

    logpriors = []
    loglikelihoods = []

    dx = 1/np.prod(F.shape)

    delF = np.gradient(F, 1/F.shape[0], 1/F.shape[1])
    # Normalize since the image covers the square around [0, 1].
    
    imdef = ag.util.DisplacementFieldWavelet(F.shape, penalty=penalty, rho=rho, wavelet=wavelet)
    
    #imdef.u[0,0,0,0,0] = 3.528
    
    x, y = imdef.meshgrid()

    # 1. 

    last_loglikelihood = -np.inf 
    last_cost = np.inf

    num_iterations = 0
    iterations_per_level = []
    if last_level is None:
        last_level = imdef.levels

    adjust = 1.0
    for a in range(start_level, last_level+1):
        ag.info("Running coarse-to-fine level", a)
        for loop_inner in xrange(max_iterations_per_level): # This number is just a maximum
            num_iterations += 1
            # 2.
        
            #print "u0 = ", imdef.u[0,0,0,0,0]

            # Calculate deformed xs
            Ux, Uy = imdef.deform_map(x, y)
            z0 = x + Ux
            z1 = y + Uy

            # Interpolate F at zs
            Fzs = ag.util.interp2d(z0, z1, F)
            #Fzs = imdef.deform(F)
            #np.testing.assert_array_almost_equal(ag.util.interp2d(z0, z1, F), Fzs)

            # Interpolate delF at zs 
            delFzs = np.empty((2,) + F.shape) 
            for q in range(2):
                delFzs[q] = ag.util.interp2d(z0, z1, delF[q], fill_value=0.0)

            # 4.
            terms = Fzs - I

            # Calculate (cost) 
    
            #sigma = np.sqrt(dx) / np.sqrt(2)
            #sigma2 = sigma**2
            # Technicall we should have dx and sigma involved here,
            # but we are letting them be absorbed into the ratio
            # between prior and loglikelihood.
            loglikelihood = -(terms**2).sum() * dx / 2.0#/2.0# * dx / (2 * sigma2)
            if calc_costs:
                logprior = imdef.logprior(a)
                logpriors.append(logprior)

                loglikelihoods.append(loglikelihood)
        
                if __debug__:# and loop_inner%10 == 0:
                    ag.info("cost = {0} (prior: {1}, llh: {2}) [max U: {3}".format(-logprior-loglikelihood, logprior, loglikelihood, (imdef.lmbks * imdef.u**2).max()))

            cost = -logprior - loglikelihood
            
            # Check termination
            #if cost > last_cost:
            #    adjust *= 2.0
            #    #ag.warning("The cost increased from {0} to {1}.".format(last_cost, cost))
            #elif adjust > 1:
            #    adjust /= 2.0

            #print np.fabs(loglikelihood - last_loglikelihood)/last_loglikelihood
            if math.fabs(cost - last_cost)/last_cost < tol and loop_inner > 5:
                break

            #print 'cost', cost 
            #print 'llh', loglikelihood
            #print 'lprior', logprior
            #print np.exp(loglikelihood)

            last_cost = cost
            last_loglikelihood = loglikelihood

            # 4.1/2 Decide step size (this is done once for each coarse-to-fine level)
            if loop_inner == 0:
                delFzs2 = delFzs[0]**2 + delFzs[1]**2

                # This is the value of the psi, which turns out to be a good bound for this purpose
                M = dx 

                T = a**2 * M * delFzs2.sum() * dx + imdef.sum_of_coefficients(a)

                if 1:
                    print 'delFzs.mean()', delFzs.min(), delFzs.max()
                    print a**2 * M * delFzs2.sum() * dx, imdef.sum_of_coefficients(a)
                    print("1/T = {0} ({1})".format(1/T, T))

                dt = stepsize_scale_factor/T


            # 5. Gradient descent
            W = np.empty((2,) + terms.shape)
            for q in range(2):
                W[q] = delFzs[q] * terms

            #import pywt
            #u = pywt.wavedec(W[0][:,0], 'db1', mode='per')
            #for uu in u[1:]:
            #    uu[:] = 0.0 
            #print "SHIFT", pywt.waverec(u, 'db1', mode='per')[0]
            #print W[0][:,0].mean()
        
            if 0:
                import pylab as plt
                print delFzs[0].shape
                plt.subplot(221)
                plt.plot(delFzs[0][:,0])
                plt.subplot(222)
                plt.plot(terms[:,0])
                plt.subplot(223) 
                plt.plot(Fzs[:,0])
                plt.subplot(224)
                plt.plot(W[0][:,0])
                plt.show()
                import sys; sys.exit(0)
                
            imdef.reestimate(dt, W, a)

        iterations_per_level.append(num_iterations)
        num_iterations = 0
    
               
    info = {}
    info['iterations_per_level'] = iterations_per_level
    info['iterations'] = sum(iterations_per_level) 
    if calc_costs:
        logpriors = np.asarray(logpriors)
        loglikelihoods = np.asarray(loglikelihoods)
        info['logpriors'] = logpriors
        info['loglikelihoods'] = loglikelihoods
        info['costs'] = -logpriors - loglikelihoods

    return imdef, info 
