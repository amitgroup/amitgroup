from __future__ import division

import math
import numpy as np
import amitgroup as ag
import amitgroup.util
import pywt

__all__ = ['imagedef']

def imagedef(F, I, A=None, penalty=0.1, rho=1.0, tol=0.01, calc_costs=False, stepsize=None, wavelet='db2', \
             max_iterations_per_level=1000, start_level=1):
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
    penalty : float
        Determines the weight of the prior (proportional to the inverse variance of the gaussian deformations). Reduce this value if you want more deformations.
    rho : float
        Determines the penalty of more granular coefficients. Increase to smoothen. Must be strictly positive.
    calc_costs : bool
        If True, then ``info`` will contain `logprior` and `loglikelihood`. The cost function `J` is the negative sum of these two. 
    stepsize : float or None
        Gradient descent step size. If `None`, then the stepsize will be inferred by something along the lines of Newton's method.
    
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
    assert rho > 0, "Parameter rho must be strictly positive"
    logpriors = []
    loglikelihoods = []

    dx = 1/np.prod(F.shape)

    # TODO: VERY TEMP
    #penalty = 1.0 
    theta = 1.0 # precision of likelihood

    delF = np.gradient(F, 1/F.shape[0], 1/F.shape[1])
    # Normalize since the image covers the square around [0, 1].
    
    imdef = ag.util.DisplacementFieldWavelet(F.shape, penalty=penalty, rho=rho, wavelet=wavelet)
    
    imdef.u[0,0,0,0,0] = 3.528
    
    x0, x1 = imdef.get_x(F.shape)

    # 1. 

    last_loglikelihood = -np.inf 
    last_cost = np.inf

    num_iterations = 0
    iterations_per_level = []
    if A is None:
        A = imdef.levels

    adjust = 1.0
    for a in range(start_level, A+1):

        ag.info("Running coarse-to-fine level", a)
        for loop_inner in xrange(max_iterations_per_level): # This number is just a maximum
            num_iterations += 1
            # 2.
        
            print "u0 = ", imdef.u[0,0,0,0,0]

            # Calculate deformed xs
            Ux, Uy = imdef.deform_map(x0, x1)
            z0 = x0 + Ux
            z1 = x1 + Uy

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
    
            print "KUU", (terms**2).sum() * dx

            #sigma = np.sqrt(dx) / np.sqrt(2)
            #sigma2 = sigma**2
            # Technicall we should have dx and sigma involved here,
            # but we are letting them be absorbed into the ratio
            # between prior and loglikelihood.
            loglikelihood = -(terms**2).sum() * dx * theta / 2.0#/2.0# * dx / (2 * sigma2)
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

            print 'cost', cost 
            print 'llh', loglikelihood
            print 'lprior', logprior
            #print np.exp(loglikelihood)

            last_cost = cost
            last_loglikelihood = loglikelihood

            # 4.1/2 Decide step size (this is done for each coarse-to-fine level)
            if loop_inner == 0 or 1:
                if stepsize is None:
                    print 'delFzs.mean()', delFzs.min(), delFzs.max()
                    delFzs2 = delFzs[0]**2 + delFzs[1]**2
                    delF2 = delF[0]**2 + delF[1]**2
                    print 'diff', delFzs2.sum(), delF2.sum()
                    
                    M = 1.0 # Upper bound of the wavelet basis functions (psi) 
                    d = imdef.lmbks[0,:a].count() 
                    print "HESS", delFzs2.sum() * dx
                    #print d, a**2
                    T = a**2 * M * theta * delFzs2.sum() * dx * dx + imdef.sum_of_coefficients(a)
                    #T /= penalty #adjust
                    #T *= 10000.0
                    print a**2 * M * delFzs2.sum() * dx, imdef.sum_of_coefficients(a)
                    print("1/T = {0} ({1})".format(1/T, T))

                    import sys; sys.exit(0)
                    delta = 1.0/T
                else:
                    delta = stepsize


            # 5. Gradient descent
            W = np.empty((2,) + terms.shape)
            for q in range(2):
                W[q] = delFzs[q] * terms * theta# * dx

            import pywt
            u = pywt.wavedec(W[0][:,0], 'db1', mode='per')
            for uu in u[1:]:
                uu[:] = 0.0 
            print "SHIFT", pywt.waverec(u, 'db1', mode='per')[0]
            print W[0][:,0].mean()
        
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
                
            imdef.reestimate(delta, W, a)

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
