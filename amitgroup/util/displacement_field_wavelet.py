from __future__ import absolute_import
from __future__ import division

import amitgroup as ag
import numpy as np
import pywt
from copy import deepcopy
from .displacement_field import DisplacementField
from .interp2d import interp2d

from amitgroup.util import wavelet

# TODO: Move these functions somewhere.
func = wavelet.wavedec2_factory((32, 32), levels=3)
invfunc = wavelet.waverec2_factory((32, 32), levels=3)

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
        This is only needed if the derivative is needed.
    rho : float
        A high value penalizes the prior for higher coarse-to-fine coefficients more.
        This is only needed if the derivative is needed.
    """
    def __init__(self, shape, wavelet='db2', rho=2.0, penalty=1.0, means=None, variances=None, level_capacity=None):
        #super(DisplacementFieldWavelet, self).__init__(shape)
        
        self.wavelet = wavelet 
        self.mode = 'per'
        self.shape = shape
        self.prepare_shape()
        self.rho = rho 
        biggest = self.scriptNs[-1]        
        self.level_capacity = level_capacity or self.levels
        N = 2**self.level_capacity 
        self.ushape = (2, N, N)

        # We divide the penalty, since the raw penalty is the ratio
        # of the variance between the coefficients and the loglikelihood.
        # It is more natural to want the variance between how much the 
        # the coefficients can create a deformation in space instead, which
        # implies an adjustment of 2**self.levels for the s.d. We take
        # the square of this since we're dealing with the variance. 
        # Notice: Penalty is only applicable if means and variances are not set manually
        if penalty:
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
    def shape_for_size(cls, size, level_capacity=np.inf):
        N = 2**level_capacity
        return (2, N, N)

    def _init_u(self):
        self.u = np.copy(self.mu)

    def _init_default_lmbks(self):
        self.lmbks = np.zeros(self.ushape)
        for level in range(self.levels+1)[::-1]:
            N = 2**level
            self.lmbks[:,:N,:N] = self.penalty_adjusted * 2.0**(self.rho * (self.scriptNs[level]-1))

    def set_flat_u(self, flat_u, level):
        """
        Sets `u` from a flattened array of a subset of `u`.
        
        The size of the subset is determined by level. The rest of `u` is filled with zeros.
        """
        assert level <= self.level_capacity, "Please increase coefficient capacity for this level"
        # First reset
        self.u.fill(0.0)
        #shape = self.coef_shape(level)
        N = 2**level
        self.u[:,:N,:N] = flat_u.reshape((2, N, N))

    def prepare_shape(self):
        side = max(self.shape)
        self.levels = int(np.log2(side))
        self.levelshape = tuple(map(int, map(np.log2, self.shape)))
        self.scriptNs = map(len, pywt.wavedec(np.zeros(side), self.wavelet, level=self.levels, mode=self.mode))

    def deform_x(self, x0, x1, last_level=np.inf):
        last_level = min(last_level, self.level_capacity)
        Ux0, Ux1 = self.invtransform(x0, x1, last_level)
        return x0+Ux0, x1+Ux1

    def deform_map(self, x, y, last_level=np.inf):
        last_level = min(last_level, self.level_capacity)
        return self.invtransform(x, y, last_level) 

    def transform(self, f, level):
        """
        Forward transform of the wavelet.
        """ 
        #old = np.asarray([
        #    self.pywt2array(pywt.wavedec2(f[q], self.wavelet, mode=self.mode, level=self.levels), self.levelshape, level, self.level_capacity) for q in range(2)
        #])
        #new = np.asarray([
        #    func(f[q], level) for q in range(2)
        #    ag.util.wavelet.new2old(func(f[q], level)) for q in range(2)
        #])
        
        new = np.zeros(self.ushape)
        new[0] = func(f[0], level)    
        new[1] = func(f[1], level)
        
        #np.testing.assert_array_almost_equal(old, new)
        return new 

    # TODO: last_level not used
    def invtransform(self, x, y, last_level=np.inf):
        """See :func:`DisplacementField.deform_map`"""

        Ux = invfunc(self.u[0], self.levels)
        Uy = invfunc(self.u[1], self.levels)
        return Ux, Uy 

    def deform(self, F, levels=np.inf):
        """See :func:`DisplacementField.deform`"""
        im = np.zeros(F.shape)

        x0, x1 = self.meshgrid()
        z0, z1 = self.deform_x(x0, x1, levels)
        im = interp2d(z0, z1, F)
        return im
    
    def abridged_u(self, last_level=None):
        #return self.u[:,:self.flat_limit(last_level)]
        N = 2**(last_level)
        return self.u[:,:N,:N]

    def coef_shape(self, last_level=None):
        return (self.ushape[0], self.flat_limit(last_level))

    def logprior(self, last_level=None):
        N = None if last_level is None else 2**last_level
        return -(self.lmbks * (self.u - self.mu)**2).reshape(2, 8, 8)[:,:N,:N].sum() / 2

    def logprior_derivative(self, last_level=None):
        N = None if last_level is None else 2**last_level
        ret = (-self.lmbks * (self.u - self.mu))[:,:N,:N]
        return ret

    def sum_of_coefficients(self, last_level=None):
        # Return only lmbks[0], because otherwise we'll double-count every
        # value (since they are the same)
        return self.lmbks[0,:self.flat_limit(last_level)].sum()

    def number_of_coefficients(self, levels=None):
        return self.ushape[1]

    def copy(self):
        return deepcopy(self) 

    def flat_limit(self, last_level=None):
        # TODO: Come up with better name, and maybe place
        return None if last_level is None else _flat_start(last_level+1, 0, self.levelshape)

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

    if 0: # TODO: Recreate
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
            if level == last_level:
                break

    # TODO: The name 'u' for the coefficients is congruent with the book, 
    #  but a bit confusing for other people. Change.
    def ulevel(self, level):
        alphas = 1 if level == 0 else 3
        size = _levels2shape(self.levelshape, level)
        #TODO: return self.u[:,level,:alphas,:size[0],:size[1]]

    def lmbk_level(self, level):
        alphas = 1 if level == 0 else 3
        size = _levels2shape(self.levelshape, level)
        return self.lmbks[:,level,:alphas,:size[0],:size[1]]

    if 0:
        @classmethod
        def pywt2array(cls, coefficients, levelshape, last_level=np.inf, level_capacity=np.inf):
            N = _total_length(levelshape, level_capacity)
            u = np.zeros(N)
            pos = 0
            for i in range(min(level_capacity+1, last_level+1, max(levelshape)+1)):
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
        def array2pywt(cls, u, levelshape, last_level):
            coefficients = []
            pos = 0
            lenU = len(u)
            for level in range(last_level+1): 
                alphas, N, M = _levels2fullshape(levelshape, level)
                sh = _levels2shape(levelshape, level)
                L = _flat_length_one_alpha(levelshape, level)
                if level == 0:  
                    coefficients.append(u[pos:pos+L].reshape(sh))
                    pos += L
                else:
                    als = []
                    for alpha in range(3):
                        if pos+L <= lenU:
                            u_segment = u[pos:pos+L]
                            als.append(u_segment.reshape(sh))
                        else:
                            als.append(np.zeros(sh))
                        pos += L
                    coefficients.append(tuple(als))
            return coefficients

