
from __future__ import division

import numpy as np
import amitgroup as ag
import amitgroup.util
import amitgroup.features
import math
import pywt
import sys
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b, fmin_cg # TODO: Remove?
from scipy.optimize.linesearch import line_search_wolfe1

def _cost(u, imdef, F, X, delFjs, x, y, a, all_js):
    """Calculate the cost."""
    if u.ndim == 1: 
        u = u.reshape((2, len(u)//2))
    imdef.u *= 0.0
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
    loglikelihood = (X * np.log(Fjzs) + (1-X) * np.log(1-Fjzs)).sum()

    # cost
    return -logprior - loglikelihood

def _cost_num_deriv(u, imdef, F, X, delFjs, x, y, level, all_js):
    """Numerical derivative for the cost. Can be used for comparison."""
    if u.ndim == 1: 
        u = u.reshape((2, len(u)//2))
    imdef.u *= 0.0
    imdef.u[:u.shape[0],:u.shape[1]] = u 

    orig_u = np.copy(imdef.u)
    
    deriv = np.zeros(orig_u.shape)
    limit = imdef.flat_limit(level) 
    dt = 0.00001
    for q in range(2):
        for i in range(limit):
            u = np.copy(orig_u)
            u[q,i] -= dt
            cost0 = _cost(u, imdef, F, X, delFjs, x, y, level, all_js)
            u = np.copy(orig_u)
            u[q,i] += dt
            cost1 = _cost(u, imdef, F, X, delFjs, x, y, level, all_js)
            deriv[q,i] = (cost1-cost0)/(2*dt)

    # Compare
    deriv = deriv[:,:limit].flatten()
    return deriv

def _cost_deriv(u, imdef, F, X, delFjs, x, y, level, all_js):
    """Calculate the derivative of the cost."""
    if u.ndim == 1:
        u = u.reshape((2, len(u)//2))
    imdef.u *= 0.0
    imdef.u[:u.shape[0],:u.shape[1]] = u 
    # Calculate deformed xs
    z0, z1 = imdef.deform_x(x, y, level)

    # Interpolate F at zs
    Fjzs = np.empty(F.shape) 
    for j in all_js: 
        Fjzs[j] = ag.util.interp2d(z0, z1, F[j])

    # Interpolate delF at zs 
    delFjzs = np.empty((2, 8) + F.shape[1:]) 
    for q in range(2):
        for j in all_js: 
            delFjzs[q,j] = ag.util.interp2d(z0, z1, delFjs[j][q], fill_value=0.0)

    # Adjusted derivatives
    """
    shape = Fjzs[0].shape
    delFjzs2 = np.empty((2, 8) + F.shape[1:])
    for j in all_js:
        grad = np.gradient(Fjzs[j], 1/shape[0], 1/shape[1])
        for q in range(2):
            delFjzs2[q,j] = grad[q]
    delFjzs = delFjzs2
    """

    W = np.empty((2,) + x.shape) # Change to empty
    for q in range(2):
        grad = delFjzs[q]
        W[q] = -((X/Fjzs - (1-X)/(1-Fjzs)) * grad).sum(axis=0) 

    limit = imdef.flat_limit(level) 
    vqks = imdef.transform(W, level)
    return (-imdef.logprior_derivative() + vqks)[:,:limit].flatten()

class _AbortException(Exception):
    pass

def bernoulli_deformation(F, I, last_level=None, penalty=1.0, gtol=0.4, rho=2.0, wavelet='db2', maxiter=1000, start_level=1, debug_plot=False, means=None, variances=None):
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
                raise _AbortException() 
            for j in range(8):
                plw.imshow(imdef.deform(F[j]), subplot=j*2)
                plw.imshow(I[j], subplot=j*2+1)
    else:
        cb = None 

    imdef = ag.util.DisplacementFieldWavelet(F.shape[1:], penalty=penalty, wavelet=wavelet, rho=rho, means=means, variances=variances)

    ### Do some tests with the derivative
    args = (imdef, F, X, delFjs, x, y, 1, all_js)
    def ccost(u):
        return _cost(u, *args)
    def ccost_num_deriv(u):
        return _cost_num_deriv(u, *args)
    def ccost_deriv(u):
        return _cost_deriv(u, *args)
    def new_u():
        u = np.zeros(imdef.u.shape)
        u[0,0] = 0.0
        return u

    if 0:
        u = new_u()
        u[0,0] = -4.0
        cc = ccost_num_deriv(u)

        print "cost:", ccost(u)
        print "deriv:", ccost_deriv(u)
        print "numde:", cc 
        import sys; sys.exit(0)

    if 0:
        print "Derv:", ccost_deriv(new_u())
        for dt in [-0.001, 0.001]:
            u = new_u() 
            cost0 = ccost(u)
                
            u = new_u() 
            u[0,0] += dt
            cost1a = ccost(u)

            u = new_u() 
            u[1,0] += dt
            cost1b = ccost(u)

            u = new_u()
            u[:,0] += dt
            cost2 = ccost(u)

            print cost1b, cost0
            numd = [(cost1a-cost0)/dt, (cost1b-cost0)/dt]
            print "dt:", dt
            print "Numd:", numd
            print "Numd2:", ccost_num_deriv(new_u())
            print "Both:", (cost2-cost0)/dt, np.sum(numd)
        import sys; sys.exit(0)

    ###

    min_cost = np.inf
    for a in range(start_level, last_level+1): 
        ag.info("Running coarse-to-fine level", a)

        u = imdef.abridged_u(a)

        if 0:
            u = u.flatten()
            args = (imdef, F, X, delFjs, x, y, a, all_js)
            def ccost(u):
                return _cost(u, *args)
            def ccost_num_deriv(u):
                return _cost_num_deriv(u, *args)
            def ccost_deriv(u):
                return _cost_deriv(u, *args)

            Hk = np.eye(len(u))
            gfk = ccost_deriv(u)
            pk = -np.dot(Hk, gfk)
            old_fval = ccost(u)
            old_old_fval = old_fval + 5000
            print 'old_fval:', old_fval
            print 'old_old_fval:', old_old_fval
            print pk
            ret = line_search_wolfe1(ccost, ccost_num_deriv, u, pk, gfk, old_fval, old_old_fval)

            print ret
            import sys; sys.exit(0)

        try:
            new_u, cost, min_deriv, Bopt, func_calls, grad_calls, warnflag = fmin_bfgs(_cost, u, _cost_deriv, args=(imdef, F, X, delFjs, x, y, a, all_js), callback=cb, gtol=gtol, maxiter=maxiter, full_output=True, disp=False)
        except _AbortException:
            return None, {}
    
        print warnflag, cost, (min_deriv.min(), min_deriv.max())

        if cost < min_cost:
            # If the algorithm makes mistakes and returns a really high cost, don't use it.
            min_cost = cost
            imdef.u[:,:u.shape[1]] = new_u.reshape(u.shape)

    #if debug_plot:
    #    plw.mainloop()

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
