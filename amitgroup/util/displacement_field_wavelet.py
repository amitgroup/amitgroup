from __future__ import absolute_import
from __future__ import division

import amitgroup as ag
import numpy as np
import pywt
from copy import deepcopy
from .displacement_field import DisplacementField
from .interp2d import interp2d

def _levels2shape(levelshape, levels, level=np.inf):
    level = min(level, levels)
    return tuple([2**(max(0, level + levelshape[q] - levels - 1)) for q in range(2)])

#TODO Move somewhere else, so as not to clog up the space
# before the class.
def _pywt2array(coefficients, levels, levelshape, maxL=np.inf):
    shape = _levels2shape(levelshape, levels)
    new_u = np.zeros((levels+1, 3) + shape)
    for i in range(min(maxL, levels+1)):
        if i == 0:
            Nx, Ny = coefficients[i].shape
            new_u[i,0,:Nx,:Ny] = coefficients[i]
        else:
            for alpha in range(3):
                Nx, Ny = coefficients[i][alpha].shape
                new_u[i,alpha,:Nx,:Ny] = coefficients[i][alpha]
                
    return new_u

def _array2pywt(coefficients, levelshape, levels):
    new_u = []
    for level in range(levels+1): 
        N, M = _levels2shape(levelshape, levels, level)
        if level == 0:
            new_u.append(coefficients[level,0,:N,:M])
        else:
            als = []
            for alpha in range(3):
                als.append(coefficients[level,alpha,:N,:M])
            new_u.append(tuple(als))
    #print map(np.shape, new_u)
    return new_u

class DisplacementFieldWavelet(DisplacementField):
    """
    Displacement field using wavelets.
    
    This class requires the package PyWavelets_.
    
    Refer to :class:`DisplacementField` for interface documentation.

    Parameters
    ----------
    shape : tuple
        Size of the displacement field.
    wavelet : string / pywt.Wavelet
        Specify wavelet type. Read more at PyWavelets_.
    penalty : float
        Coefficient signifying the size of the prior. Higher means less deformation.
        This is only needed if this deformation will be reestimated.
    rho : float
        A high value penalizes the prior for higher coarse-to-fine coefficients more. Must be strictly greater than 0.
        This is only needed if this deformation will be reestimated.
     
    .. _PyWavelets: http://www.pybytes.com/pywavelets/
    """
    def __init__(self, shape, wavelet='db2', rho=2.0, penalty=1.0, means=None, variances=None):
        assert rho > 0.0, "Parameter rho must be strictly positive"
        #super(DisplacementFieldWavelet, self).__init__(shape)
        self.wavelet = wavelet 
        self.mode = 'per'
        self.shape = shape
        self.prepare_shape()
        self.rho = rho 
        biggest = self.scriptNs[-1]        
        coefshape = _levels2shape(self.levelshape, self.levels)
        self.ushape = (2, self.levels+1, 3) + coefshape
        
        # We divide the penalty, since the raw penalty is the ratio
        # of the variance between the coefficients and the loglikelihood.
        # It is more natural to want the variance between how much the 
        # the coefficients can create a deformation in space instead, which
        # implies an adjustment of 2**self.levels for the s.d. We take
        # the square of this since we're dealing with the variance. 
        # Notice: Penalty is only applicable
        self.penalty_adjusted = penalty / 4**self.levels 

        self._init_mask()

        if means is not None:
            self.mu = np.ma.array(means, mask=self.mask)
        else:
            self.mu = np.ma.array(np.zeros(self.ushape), mask=self.mask)

        if variances is not None:
            self.lmbks = np.ma.array(1/np.clip(variances, 0.0001, 1000000), mask=self.mask)
        else:
            self._init_default_lmbks()

        self._init_u()

    @classmethod
    def shape_for_size(cls, size):
        levelshape = tuple(map(int, map(np.log2, size)))
        levels = max(levelshape)
        coefshape = _levels2shape(levelshape, levels)
        return (2, levels+1, 3) + coefshape
            
    def _init_mask(self):
        self.mask = np.ones(self.ushape)
        for level in range(self.levels+1):
            N, M = _levels2shape(self.levelshape, self.levels, level)
            if level == 0:
                self.mask[:,level,0,:N,:M] = 0 
            else:
                self.mask[:,level,:,:N,:M] = 0 

    def _init_u(self):
        # Could also default to mean values
        self.u = np.ma.array(np.zeros(self.ushape), mask=self.mask)

    def _init_default_lmbks(self):
        values = np.zeros(self.ushape)
        for i in range(self.levels+1):
            # We decrease the self.scriptNs[i] so that the first level
            # is only the penalty
            values[:,i,:,:,:] = self.penalty_adjusted * 2.0**(self.rho * (self.scriptNs[i]-1))

        self.lmbks = np.ma.array(values, mask=self.mask)

    def prepare_shape(self):
        side = max(self.shape)
        self.levels = int(np.log2(side))
        self.levelshape = tuple(map(int, map(np.log2, self.shape)))
        self.scriptNs = map(len, pywt.wavedec(np.zeros(side), self.wavelet, level=self.levels, mode=self.mode))

    def _deformed_x(self, x0, x1):
        Ux0, Ux1 = self.deform_map(x0, x1)
        return x0+Ux0, x1+Ux1

    def deform_map(self, x, y):
        """See :func:`DisplacementField.deform_map`"""
        # TODO: Do waverec2 with cutoff coefficients and then patch it up with
        # linear interpolation instead! Should give comparable results, at least
        # for db2, but possibly faster.
    
        defx0 = pywt.waverec2(_array2pywt(self.u[0], self.levelshape, self.levels), self.wavelet, mode=self.mode) 
        defx1 = pywt.waverec2(_array2pywt(self.u[1], self.levelshape, self.levels), self.wavelet, mode=self.mode)

        # Interpolated defx at xs 
        if x.shape == defx0.shape:
            Ux = defx0
            Uy = defx1
        else:
            Ux = interp2d(x, y, defx0, dx=np.array([1/(defx0.shape[0]-1), 1/(defx0.shape[1]-1)]))
            Uy = interp2d(x, y, defx1, dx=np.array([1/(defx1.shape[0]-1), 1/(defx1.shape[1]-1)]))

        return Ux, Uy 

    def deform(self, F):
        """See :func:`DisplacementField.deform`"""
        im = np.zeros(F.shape)

        x0, x1 = self.meshgrid()
        z0, z1 = self._deformed_x(x0, x1)
        im = interp2d(z0, z1, F)
        return im

    def logprior(self, levels=None):
        return -(self.lmbks * (self.u - self.mu)**2)[:,:levels].sum() / 2

    def reestimate(self, stepsize, W, level):
        """
        Reestimation step for training the deformation. 
        """
        vqks = np.asarray([
            _pywt2array(pywt.wavedec2(W[q], self.wavelet, mode=self.mode, level=self.levels), self.levels, self.levelshape, level) for q in range(2)
        ]) / 4**self.levels # Notice this adjustment of the values

        self.u -= stepsize * (self.lmbks * self.u + vqks)

    def sum_of_coefficients(self, levels=None):
        # Return only lmbks[0], because otherwise we'll double-count every
        # value (since they are the same)
        # TODO: Just changed this. Correct?
        return self.lmbks[0,:levels].sum()

    def number_of_coefficients(self, levels=None):
        return (self.mask[:,:levels] == False).sum()

    def copy(self):
        return deepcopy(self) 

    def randomize(self, sigma, rho=2.5, start_level=1, levels=3):
        """
        Randomly sets the coefficients up to a certain level by sampling a Gaussian. 
        
        Parameters
        ----------  
        sigma : float
            Standard deviation of the Gaussian. The `sigma` is adjusted to a normalized image
            scale and not the scale of coefficient values (nor pixels). This means that setting `sigma` to 1, the standard
            deviation is the same size as the image, which is a lot. A more appropriate value is
            thus 0.01.
        rho : float
            A value higher than 1, will cause more dampening for higher coefficients, which will
            result in a smoother deformation.
        levels: int
            Number of levels that should be randomized. The levels above will be set to zero. For a funny-mirror-type deformation, this should be limited to about 3.
        """
        # Reset all values first
        self.u *= 0.0
    
        for q in range(2):
            for level in range(start_level, min(self.levels+1, start_level+levels)):
                N, M = _levels2shape(self.levelshape, self.levels, level)

                # First of all, a coefficient of 1, will be shift the image 1/2**self.levels, 
                # so first we have to adjust for that.
                # Secondly, higher coefficient should be adjusted by roughly 2**-s, to account
                # for the different amplitudes of a wavelet basis (energy-conserving reasons).
                # Finally, we might want to dampen higher coefficients even further, to create
                # a smoother image. This is done by rho.
                adjust = 2.0**(self.levels - rho * max(level-1, 0))# * 2**self.levels

                if level == 0:
                    self.u[q,level,0,:N,:M] = np.random.normal(0.0, sigma, (N, M)) * adjust 
                else:
                    als = []
                    for alpha in range(3):
                        self.u[q,level,alpha,:N,:M] = np.random.normal(0.0, sigma, (N, M)) * adjust 

    def ilevels(self):
        for level in range(self.levels+1):
            alphas = 1 if level == 0 else 3
            yield level, (alphas,)+_levels2shape(self.levelshape, self.levels, level)

    def print_lmbks(self, last_level=np.inf):
        for level, (alphas, N, M) in self.ilevels():
            print "Level {0}".format(level)
            print self.lmbks[:,level,:alphas,:N,:M]
            if level == last_level:
                break

    def print_u(self, last_level=np.inf):
        for level, (alphas, N, M) in self.ilevels():
            print "Level {0}".format(level)
            print self.u[:,level,:alphas,:N,:M]
            if level == last_level:
                break

    # TODO: The name 'u' for the coefficients is congruent with the book, 
    #  but a bit confusing for other peopel. Change.
    def ulevel(self, level):
        alphas = 1 if level == 0 else 3
        size = _levels2shape(self.levelshape, self.levels, level)
        return self.u[:,level,:alphas,:size[0],:size[1]]

    def lmbk_level(self, level):
        alphas = 1 if level == 0 else 3
        size = _levels2shape(self.levelshape, self.levels, level)
        return self.lmbks[:,level,:alphas,:size[0],:size[1]]
