from __future__ import absolute_import
from __future__ import division

import amitgroup as ag
import numpy as np
import pywt
from copy import deepcopy
from .displacement_field import DisplacementField
from .interp2d import interp2d

#TODO Move somewhere else, so as not to clog up the space
# before the class.
def _pywt2array(coefficients, scriptNs, maxL=np.inf):
    L = len(scriptNs)#len(coefficients)
    N = scriptNs[-1]#len(coefficients[-1][0])
    new_u = np.zeros((L, 3, N, N))
    for i in range(min(maxL, L)):
        if i == 0:
            Nx, Ny = coefficients[i].shape
            assert Nx == scriptNs[i]
            new_u[i,0,:Nx,:Ny] = coefficients[i]
        else:
            for alpha in range(3):
                Nx, Ny = coefficients[i][alpha].shape
                assert Nx == scriptNs[i]
                new_u[i,alpha,:Nx,:Ny] = coefficients[i][alpha]
                
    return new_u

def _array2pywt(coefficients, scriptNs):
    new_u = []
    for i, N in enumerate(scriptNs): 
        if i == 0:
            new_u.append(coefficients[i,0,:N,:N])
        else:
            als = []
            for alpha in range(3):
                als.append(coefficients[i,alpha,:N,:N])
            new_u.append(tuple(als))
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
    penalty : float
        Coefficient signifying the size of the prior. Higher means less deformation.
    rho : float
        A high value penalizes the prior for higher coarse-to-fine coefficients more. Must be strictly greater than 0.
    wavelet : string / pywt.Wavelet
        Specify wavelet type. Read more at PyWavelets_.
     
    .. _PyWavelets: http://www.pybytes.com/pywavelets/
    """
    def __init__(self, shape, penalty=1.0, rho=1.0, wavelet='db2'):
        assert rho > 0.0, "Parameter rho must be strictly positive"
        #super(DisplacementFieldWavelet, self).__init__(shape)
        self.wavelet = wavelet 
        self.mode = 'per'
        self.shape = shape
        self.prepare_shape()
        self.penalty = penalty
        self.rho = rho 
        biggest = self.scriptNs[-1]        
        self.ushape = (2, self.levels+1, 3, biggest, biggest)
        #self.u = np.zeros(self.ushape)
        self._init_lmbks_and_u()

    def _init_lmbks_and_u(self):
        values = np.zeros(self.ushape)
        for i in range(self.levels+1):
            # We decrease the self.scriptNs[i] so that the first level
            # is only the penalty
            values[:,i,:,:,:] = self.penalty * 2.0**(self.rho * self.scriptNs[i]) / 2.0
            # * np.prod(self.shape)

        # Which ones are used? Create a mask
        mask = np.ones(self.ushape)
        for level in range(self.levels+1):
            N = 2**(max(0, level-1))
            if level == 0:
                mask[:,level,0,:N,:N] = 0 
            else:
                mask[:,level,:,:N,:N] = 0 

        self.lmbks = np.ma.array(values, mask=mask)
        self.u = np.ma.array(np.zeros(self.ushape), mask=mask)

    def prepare_shape(self):
        #self.levels = len(pywt.wavedec(range(shape[0]), wl)) - 1
        self.levels = int(np.log2(self.shape[0]))
        self.scriptNs = map(len, pywt.wavedec(range(self.shape[0]), self.wavelet, level=self.levels, mode=self.mode))

    def _deformed_x(self, x0, x1):
        Ux0, Ux1 = self.deform_map(x0, x1)
        return x0+Ux0, x1+Ux1

    def deform_map(self, x, y):
        """See :func:`DisplacementField.deform_map`"""
        # TODO: Do waverec2 with cutoff coefficients and then patch it up with
        # linear interpolation instead! Should give comparable results, at least
        # for db2.
        defx0 = pywt.waverec2(_array2pywt(self.u[0], self.scriptNs), self.wavelet, mode=self.mode) 
        defx1 = pywt.waverec2(_array2pywt(self.u[1], self.scriptNs), self.wavelet, mode=self.mode)

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

        x0, x1 = self.get_x(F.shape) 
        z0, z1 = self._deformed_x(x0, x1)
        im = interp2d(z0, z1, F)
        return im

    def logprior(self, levels=None):
        return -(self.lmbks * self.u**2)[:,:levels].sum() / 2

    def logprior_x2(self, levels=None):
        return -(self.lmbks * self.u**2)[:,:levels].sum()

    def reestimate(self, stepsize, W, level):
        """
        Reestimation step for training the deformation. 
        """
        vqks = np.asarray([
            _pywt2array(pywt.wavedec2(W[q], self.wavelet, mode=self.mode, level=self.levels), self.scriptNs, level) for q in range(2)
        ])


        if 0:
            quit = self.u[0,0,0,0,0] != 0
            print "HERE", self.lmbks[0,0,0,0,0], vqks[0,0,0,0,0]
            print "LAMBDAS: ", self.lmbks[0,0,0,0,0], self.u[0,0,0,0,0]
            print "LAMBDAS: ", -self.lmbks[0,0,0,0,0] * self.u[0,0,0,0,0]
            print "VS:      ", -vqks[0,0,0,0,0]/np.prod(self.shape)
            if quit: 
                import sys; sys.exit(0)

        self.u -= stepsize * (self.lmbks * self.u + vqks / np.prod(self.shape))

    def sum_of_coefficients(self, levels=None):
        # Return only lmbks[0], because otherwise we'll double-count every
        # value (since they are the same)
        return self.lmbks[0,:levels].sum()

    def copy(self):
        return deepcopy(self) 
