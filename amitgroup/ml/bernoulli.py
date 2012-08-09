
from __future__ import division

import numpy as np
import amitgroup as ag
import amitgroup.util
import amitgroup.features
import math
import pywt

def bernoulli_deformation(F, I, last_level=None, penalty=1.0, rho=2.0, tol=0.001, \
                          wavelet='db2', max_iterations_per_level=1000, start_level=1, stepsize_scale_factor=1.0, \
                          debug_plot=False, means=None, variances=None):
    """
    Similar to :func:`image_deformation`, except it operates on binary features.
    """
    #assert F.shape[1:] == I.shape, "F and I must match"
    assert F.shape == I.shape, "F and I must match"
    assert F.shape[0] == 8, "F must already be the means of edge features"

    logpriors = []
    loglikelihoods = []
    costs = []

    # This, or an assert
    X = I.astype(float)

    all_js = range(8)

    delFjs = []
    for j in all_js:
        delF = np.gradient(F[j], 1/F[j].shape[0], 1/F[j].shape[1])
        # Normalize since the image covers the square around [0, 1].
        delFjs.append(delF)
    
    imdef = ag.util.DisplacementFieldWavelet(F.shape[1:], penalty=penalty, wavelet=wavelet, rho=rho, means=means, variances=variances)

    x, y = imdef.meshgrid()

    # 1. 
    dx = 1/np.prod(F.shape)

    last_loglikelihood = -np.inf 
    last_cost = np.inf

    num_iterations = 0
    iterations_per_level = []
    if last_level is None:
        last_level = imdef.levels

    if debug_plot:
        plw = ag.plot.PlottingWindow(figsize=(6, 10), subplots=(5, 4))

    for a in range(start_level, last_level+1): 
        ag.info("Running coarse-to-fine level", a)
        for loop_inner in xrange(max_iterations_per_level):
            if debug_plot and not plw.tick():
                break 
            num_iterations += 1
            # 2. Deform

            # Calculate deformed xs
            Ux, Uy = imdef.deform_map(x, y)
            z0 = x + Ux
            z1 = y + Uy

            # Interpolate F at zs
            Fjzs = []
            for j in all_js: 
                Fzs = ag.util.interp2d(z0, z1, F[j].astype(float))
                Fjzs.append(Fzs)

            # Interpolate delF at zs 
            delFjzs = []
            for j in all_js: 
                delFzs = np.empty((2,) + F[j].shape) 
                for q in range(2):
                    delFzs[q] = ag.util.interp2d(z0, z1, delFjs[j][q], fill_value=0.0)
                delFjzs.append(delFzs)

            # 3. Cost

            # log-prior
            logprior = imdef.logprior()
            logpriors.append(logprior)
    
            # log-likelihood
            loglikelihood = 0.0
            for j in all_js:
                loglikelihood += (X[j] * np.log(Fjzs[j]) + (1-X[j]) * np.log(1.0-Fjzs[j])).sum()
            loglikelihoods.append(loglikelihood)

            # cost
            cost = -logprior - loglikelihood
            costs.append(cost)

            if __debug__:
                ag.info("cost: {0} (prior: {1}, llh: {2})".format(cost, logprior, loglikelihood))
            
            # Some plotting for real-time feedback
            if debug_plot:
                for j in all_js:
                    # What's the likelihood for this component?
                    llhj = (X[j] * np.log(Fjzs[j]) + (1-X[j]) * np.log(1.0-Fjzs[j])).sum()
                    plw.imshow(Fjzs[j], subplot=j*2+0, caption="{0:.2f}".format(llhj))
                    plw.imshow(X[j], subplot=j*2+1)
                plw.plot(costs[-100:], subplot=16)
                plw.plot(loglikelihoods[-100:], subplot=17)
                plw.plot(logpriors[-100:], subplot=18)
                plw.imshow(np.asarray(Fjzs).mean(axis=0), subplot=19, caption="Average F (deformed)")
                plw.flip(20)

            # Check termination
            if math.fabs(cost - last_cost)/last_cost < tol and loop_inner > 5:
                break

            last_cost = cost
            last_loglikelihood = loglikelihood

            # 4. Decide step size (this is done once for each coarse-to-fine level)
            if loop_inner == 0:
                v = 0.0
                # TODO: This can be simplified greatly, either with temporary variables or actually simplifying the expression,
                # or, maybe even removing less impactful factors.
                for j in all_js: 
                    del2Fjzs = np.asarray([
                        np.gradient(delFjzs[j][q], 1/x.shape[0], 1/x.shape[1])[q] for q in range(2)
                    ]) 

                    v += (X[j] * (np.fabs((Fjzs[j] * del2Fjzs[0] * del2Fjzs[1])) + delFjzs[j][0]**2 + delFjzs[j][1]**2)/Fjzs[j]**2).sum()
                    v += ((1-X[j]) * (np.fabs((1-Fjzs[j]) * del2Fjzs[0] * del2Fjzs[1]) + delFjzs[j][0]**2 + delFjzs[j][1]**2)/(1-Fjzs[j])**2).sum()

                # This is the value of the psi, which turns out to be a good bound for this purpose
                M = dx
                T = imdef.number_of_coefficients(a) * M * v * dx + np.sqrt(imdef.sum_of_coefficients(a))# * 8

                dt = stepsize_scale_factor/T

            # 5. Gradient descent
            W = np.zeros((2,) + x.shape) # Change to empty
            for q in range(2):
                Wq = 0.0
                for j in all_js: 
                    grad = delFjzs[j][q]
                    Xj = X[j]
                    Fjzsj = Fjzs[j]

                    t1 = Xj/Fjzsj
                    t2 = -(1-Xj)/(1-Fjzsj)
                    # This erronously says plus in Amit's book.
                    Wq -= (t1 + t2) * grad
                W[q] = Wq 

            imdef.reestimate(dt, W, a+1)

        iterations_per_level.append(num_iterations)
        num_iterations = 0
    
    if debug_plot:
        plw.mainloop()
               
    info = {}
    info['iterations_per_level'] = iterations_per_level
    info['iterations'] = sum(iterations_per_level) 
    logpriors = np.asarray(logpriors)
    loglikelihoods = np.asarray(loglikelihoods)
    info['logpriors'] = logpriors
    info['loglikelihoods'] = loglikelihoods
    info['costs'] = -logpriors - loglikelihoods

    return imdef, info 
