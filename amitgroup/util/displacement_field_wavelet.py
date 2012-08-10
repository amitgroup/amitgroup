from __future__ import absolute_import
from __future__ import division

import amitgroup as ag
import numpy as np
import pywt
from copy import deepcopy
from .displacement_field import DisplacementField
from .interp2d import interp2d

def _size2levelshape(size):
    return tuple(map(int, map(np.log2, size)))

def _levels2shape(levelshape, level=np.inf):
    levels = max(levelshape)
    level = min(level, levels)
    return tuple([2**(max(0, level + levelshape[q] - levels - 1)) for q in range(2)])

# This shape include alpha
def _levels2fullshape(levelshape, level):
    sh = _levels2shape(levelshape, level)
    return ((1,) if level == 0 else (3,)) + sh 

def _total_length(levelshape):
    return _flat_start(max(levelshape)+1, 0, levelshape)

def _flat_length_one_alpha(levelshape, level):
    return np.prod(_levels2shape(levelshape, level))

def _flat_length(levelshape, level):
    return np.prod(_levels2fullshape(levelshape, level))

def _flat_start(level, alpha, levelshape):
    start = 0
    for l in range(level):
        start += _flat_length(levelshape, l)

    start += alpha * _flat_length_one_alpha(levelshape, level)
    return start    

class DisplacementFieldWavelet(DisplacementField):
    """
    Displacement field using wavelets.
    
    This class requires the package `PyWavelets <http://www.pybytes.com/pywavelets/>`_.
    
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
        coefshape = _levels2shape(self.levelshape)
        N = _total_length(self.levelshape)
        self.ushape = (2, N)

        # We divide the penalty, since the raw penalty is the ratio
        # of the variance between the coefficients and the loglikelihood.
        # It is more natural to want the variance between how much the 
        # the coefficients can create a deformation in space instead, which
        # implies an adjustment of 2**self.levels for the s.d. We take
        # the square of this since we're dealing with the variance. 
        # Notice: Penalty is only applicable
        self.penalty_adjusted = penalty / 4**self.levels 

        if means is not None:
            self.mu = means 
        else:
            self.mu = np.zeros(self.ushape)

        if variances is not None:
            self.lmbks = 1/variances
        else:
            self._init_default_lmbks()

        self._init_u()

    @classmethod
    def shape_for_size(cls, size):
        levelshape = _size2levelshape(size)
        levels = max(levelshape)
        coefshape = _levels2shape(levelshape)
        return (2, levels+1, 3) + coefshape

    def _init_u(self):
        # Could also default to mean values
        self.u = np.zeros(self.ushape)

    def _init_default_lmbks(self):
        values = np.zeros(self.ushape)
        for i in range(self.levels+1):
            # We decrease the self.scriptNs[i] so that the first level
            # is only the penalty
            n0 = _flat_start(i, 0, self.levelshape)
            n1 = n0 + _flat_length(self.levelshape, i)
            values[:,self.range_slice(i)] = self.penalty_adjusted * 2.0**(self.rho * (self.scriptNs[i]-1))

        self.lmbks = values

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

        defx0 = pywt.waverec2(self.array2pywt(self.u[0], self.levelshape, self.levels), self.wavelet, mode=self.mode) 
        defx1 = pywt.waverec2(self.array2pywt(self.u[1], self.levelshape, self.levels), self.wavelet, mode=self.mode)

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
        limit = None if levels is None else _flat_start(levels, 0, self.levelshape)
        return -(self.lmbks * (self.u - self.mu)**2)[:,:limit].sum() / 2

    def reestimate(self, stepsize, W, level):
        """
        Reestimation step for training the deformation. 
        """
        vqks = np.asarray([
            self.pywt2array(pywt.wavedec2(W[q], self.wavelet, mode=self.mode, level=self.levels), self.levels, self.levelshape, level) for q in range(2)
        ]) / 4**self.levels # Notice this adjustment of the values

        self.u -= stepsize * (self.lmbks * self.u + vqks)
    
    def derive(self, W, level):
        vqks = np.asarray([
            self.pywt2array(pywt.wavedec2(W[q], self.wavelet, mode=self.mode, level=self.levels), self.levels, self.levelshape, level) for q in range(2)
        ]) / 4**self.levels # Notice this adjustment of the values
        
        return self.lmbks * self.u + vqks

    def sum_of_coefficients(self, levels=None):
        # Return only lmbks[0], because otherwise we'll double-count every
        # value (since they are the same)
        return self.lmbks[0,:_flat_start(levels, 0, self.levelshape)].sum()

    def number_of_coefficients(self, levels=None):
        return self.ushape[1]

    def copy(self):
        return deepcopy(self) 

    def range_slice(self, level, alpha=None):
        """
        Return a slice object corresponding to the range in the flattened coefficient array.

        Parameters
        ----------
        level : int
            Coefficient level.
        alpha : int or None
            If None, then the range of all alphas are returned. Otherwise, only the specified alpha.
        """
        n0 = _flat_start(level, 0, self.levelshape)
        if alpha is None:
            n1 = n0 + _flat_length(self.levelshape, level)
        else:
            n1 = n0 + _flat_length_one_alpha(self.levelshape, level)
        return slice(n0,n1)

    def randomize(self, sigma=0.01, rho=2.5, start_level=1, levels=3):
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

        Examples
        --------
        >>> import amitgroup as ag
        >>> import matplotlib.pylab as plt

        Generate 9 randomly altered faces.
        
        >>> face = ag.io.load_example('faces')[0]
        >>> imdef = ag.util.DisplacementFieldWavelet(face.shape, 'db8')
        >>> ag.plot.images([imdef.randomize(0.1).deform(face) for i in range(9)])
        >>> plt.show()
        """
        # Reset all values first
        self.u *= 0.0
    
        for q in range(2):
            for level in range(start_level, min(self.levels+1, start_level+levels)):
                N, M = _levels2shape(self.levelshape, level)

                # First of all, a coefficient of 1, will be shift the image 1/2**self.levels, 
                # so first we have to adjust for that.
                # Secondly, higher coefficient should be adjusted by roughly 2**-s, to account
                # for the different amplitudes of a wavelet basis (energy-conserving reasons).
                # Finally, we might want to dampen higher coefficients even further, to create
                # a smoother image. This is done by rho.
                adjust = 2.0**(self.levels - rho * max(level-1, 0))# * 2**self.levels

                if level == 0:
                    self.u[q,self.range_slice(level, 0)] = np.random.normal(0.0, sigma, (n1-n0)) * adjust
                else:
                    als = []
                    for alpha in range(3):
                        n0 = _flat_start(level, 0, self.levelshape)
                        n1 = n0 + _flat_length_one_alpha(self.levelshape, level)
                        self.u[q,self.range_slice(level, alpha)] = np.random.normal(0.0, sigma, (n1-n0)) * adjust
        return self

    def ilevels(self):
        for level in range(self.levels+1):
            alphas = 1 if level == 0 else 3
            yield level, (alphas,)+_levels2shape(self.levelshape, level)

    def print_lmbks(self, last_level=np.inf):
        for level, (alphas, N, M) in self.ilevels():
            if level == last_level:
                break

    def print_u(self, last_level=np.inf):
        for level, (alphas, N, M) in self.ilevels():
            #TODO: print self.u[:,level,:alphas,:N,:M]
            print self.u
            if level == last_level:
                break

    # TODO: The name 'u' for the coefficients is congruent with the book, 
    #  but a bit confusing for other peopel. Change.
    def ulevel(self, level):
        alphas = 1 if level == 0 else 3
        size = _levels2shape(self.levelshape, level)
        #TODO: return self.u[:,level,:alphas,:size[0],:size[1]]

    def lmbk_level(self, level):
        alphas = 1 if level == 0 else 3
        size = _levels2shape(self.levelshape, level)
        return self.lmbks[:,level,:alphas,:size[0],:size[1]]

    @classmethod
    def pywt2array(cls, coefficients, levels, levelshape, maxL=np.inf):
        N = _total_length(levelshape)
        u = np.zeros(N)
        pos = 0
        for i in range(min(maxL, levels+1)):
            L = _flat_length_one_alpha(levelshape, i)
            if i == 0:
                u[pos:pos+L] = coefficients[i].flatten()
                pos += L
            else:
                for a in range(3): 
                    u[pos:pos+L] = coefficients[i][a].flatten()
                    pos += L
            
        return u

    @classmethod
    def array2pywt(cls, u, levelshape, levels):
        coefficients = []
        pos = 0
        for level in range(levels+1): 
            alphas, N, M = _levels2fullshape(levelshape, level)
            sh = _levels2shape(levelshape, level)
            L = _flat_length_one_alpha(levelshape, level)
            if level == 0:  
                coefficients.append(u[pos:pos+L].reshape(sh))
                pos += L
            else:
                als = []
                for alpha in range(3):
                    als.append(u[pos:pos+L].reshape(sh))
                    pos += L
                coefficients.append(tuple(als))
        return coefficients

