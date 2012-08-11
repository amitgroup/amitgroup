
from __future__ import division

import numpy as np
import amitgroup as ag
import amitgroup.util
import amitgroup.features
import math
import pywt
import sys
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b

def _cost(u, imdef, F, X, delFjs, x, y, a, all_js):
    if len(u.shape) == 1:
        u = u.reshape((2, len(u)//2))
    imdef.u[:u.shape[0],:u.shape[1]] = u 
    # Calculate deformed xs
    z0, z1 = imdef.deform_x(x, y, a)

    # Interpolate F at zs
    Fjzs = np.empty(F.shape) 
    for j in all_js: 
        Fjzs[j] = ag.util.interp2d(z0, z1, F[j].astype(float))

    # 3. Cost

    # log-prior
    logprior = imdef.logprior()

    # log-likelihood
    loglikelihood = (X * np.log(Fjzs) + (1-X) * np.log(1.0-Fjzs)).sum()

    # cost
    return -logprior - loglikelihood

def _cost_deriv(u, imdef, F, X, delFjs, x, y, a, all_js):
    if len(u.shape) == 1:
        u = u.reshape((2, len(u)//2))
    imdef.u[:u.shape[0],:u.shape[1]] = u 
    # Calculate deformed xs
    z0, z1 = imdef.deform_x(x, y, a)

    # Interpolate delF at zs 
    delFjzs = np.empty((8, 2) + F.shape[1:]) 
    for j in all_js: 
        for q in range(2):
            delFjzs[j,q] = ag.util.interp2d(z0, z1, delFjs[j][q], fill_value=0.0)

    # Interpolate F at zs
    Fjzs = np.empty(F.shape) 
    for j in all_js: 
        Fjzs[j] = ag.util.interp2d(z0, z1, F[j].astype(float))

    W = np.zeros((2,) + x.shape) # Change to empty
    for q in range(2):
        grad = delFjzs[:,q]
        W[q] = -((X/Fjzs - (1-X)/(1-Fjzs)) * grad).sum(axis=0) 

    limit = None if a is None else ag.util.flat_start(a, 0, imdef.levelshape)
    return imdef.derive(W, a+1)[:,:limit].flatten()

def bernoulli_deformation(F, I, last_level=None, penalty=1.0, gtol=0.4, rho=2.0, wavelet='db8', maxiter=1000, start_level=1, debug_plot=False, means=None, variances=None):
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

    if debug_plot:
        plw = ag.plot.PlottingWindow(figsize=(8, 8), subplots=(4,4))
        def cb(uk):
            if not plw.tick():
                sys.exit(0)
            for j in range(8):
                plw.imshow(imdef.deform(F[j]), subplot=j*2)
                plw.imshow(I[j], subplot=j*2+1)
    else:
        cb = None 

    imdef = ag.util.DisplacementFieldWavelet(F.shape[1:], penalty=penalty, wavelet=wavelet, rho=rho, means=means, variances=variances)

    min_cost = np.inf
    for a in range(start_level, last_level+1): 
        ag.info("Running coarse-to-fine level", a)

        u = imdef.abridged_u(a)
        new_u, cost = fmin_bfgs(_cost, u, _cost_deriv, args=(imdef, F, X, delFjs, x, y, a, all_js), callback=cb, gtol=gtol, maxiter=maxiter, full_output=True)[:2]

        #new_u, cost, _ = fmin_l_bfgs_b(_cost, u, _cost_deriv, args=(imdef, F, X, delFjs, x, y, a, all_js))
        print cost
        if cost < min_cost:
            # If the algorithm makes mistakes and returns a really high cost, don't use it.
            min_cost = cost
            imdef.u[:,:u.shape[1]] = new_u.reshape(u.shape)

    return imdef, {'cost': min_cost}
    
def _bernoulli_deformation_old(F, I, last_level=None, penalty=1.0, rho=2.0, tol=0.001, \
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
