from __future__ import absolute_import

import amitgroup as ag
import numpy as np
import pywt
from .displacement_field import DisplacementField
from .interp2d import interp2d

#TODO Move somewhere else, so as not to clog up the space
# before the class.
def _pywt2array(coef, scriptNs, maxL=1000):
    L = len(scriptNs)#len(coef)
    N = scriptNs[-1]#len(coef[-1][0])
    new_u = np.zeros((L, 3, N, N))
    for i in range(min(maxL, L)):
        if i == 0:
            Nx, Ny = coef[i].shape
            new_u[i,0,:Nx,:Ny] = coef[i]
        else:
            for alpha in range(3):
                Nx, Ny = coef[i][alpha].shape
                new_u[i,alpha,:Nx,:Ny] = coef[i][alpha]
                
    return new_u

def _array2pywt(coef, scriptNs):
    new_u = []
    for i, N in enumerate(scriptNs): 
        if i == 0:
            new_u.append(coef[i,0,:N,:N])
        else:
            als = []
            for alpha in range(3):
                als.append(coef[i,alpha,:N,:N])
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
    coef : float
        Coefficient signifying the size of the prior. Higher means less deformation.
    rho : float
        Higher value penalizes higher coarse-to-fine coefficients.
    wavelet : string / pywt.Wavelet
        Specify wavelet type. Read more at PyWavelets_.
     
    .. _PyWavelets: http://www.pybytes.com/pywavelets/
    """
    def __init__(self, shape, coef=1e-3, rho=1.5, wavelet='db2'):
        self.wavelet = wavelet 
        self.mode = 'per'
        super(DisplacementFieldWavelet, self).__init__(shape)
        self.shape = shape
        self.rho = rho 
        self.coef = coef
        biggest = self.scriptNs[-1]        
        self.ushape = (2, self.levels+1, 3, biggest, biggest)
        self.u = np.zeros(self.ushape)
        self._init_lmbks()

    def _init_lmbks(self):
        values = np.zeros(self.ushape)
        for i in range(self.levels+1):
            values[:,i,:,:,:] = self.coef * 2.0**(self.rho * i) 

        # Which ones are used? Create a mask
        mask = np.ones(self.ushape)
        for level in range(self.levels+1):
            N = 2**(max(0, level-1))
            if level == 0:
                mask[:,level,0,:N,:N] = 0 
            else:
                mask[:,level,:,:N,:N] = 0 

        self.lmbks = np.ma.array(values, mask=mask)
    

    def prepare_shape(self, shape):
        #self.levels = len(pywt.wavedec(range(shape[0]), wl)) - 1
        self.levels = int(np.log2(shape[0]))
        self.scriptNs = map(len, pywt.wavedec(range(shape[0]), self.wavelet, level=self.levels, mode=self.mode))

    def _deformed_x(self, x0, x1):
        Ux0, Ux1 = self.deform_map(x0, x1)
        return x0+Ux0, x1+Ux1

    def deform_map(self, x, y):
        """See :func:`DisplacementField.deform_map`"""
        defx0 = pywt.waverec2(_array2pywt(self.u[0], self.scriptNs), self.wavelet, mode=self.mode) 
        defx1 = pywt.waverec2(_array2pywt(self.u[1], self.scriptNs), self.wavelet, mode=self.mode)

        # Interpolated defx at xs 
        Ux = interp2d(x, y, defx0, dx=np.array([1.0/(defx0.shape[0]-1), 1.0/(defx0.shape[1]-1)]))
        Uy = interp2d(x, y, defx1, dx=np.array([1.0/(defx1.shape[0]-1), 1.0/(defx1.shape[1]-1)]))
        return Ux, Uy 

    def deform(self, F):
        """See :func:`DisplacementField.deform`"""
        im = np.zeros(F.shape)

        x0, x1 = self.get_x(F.shape) 
        z0, z1 = self._deformed_x(x0, x1)
        im = interp2d(z0, z1, F)
        return im

    def logprior(self):
        return (self.lmbks * self.u**2).sum() / 2.0

    def reestimate(self, stepsize, W, level):
        """
        Reestimation step for training the deformation. 
        """
        vqks = np.asarray([
            _pywt2array(pywt.wavedec2(W[q], self.wavelet, mode=self.mode, level=self.levels), self.scriptNs, level) for q in range(2)
        ])

        self.u -= stepsize * (self.lmbks * self.u + vqks)

    def sum_of_coefficients(self, to_level=None):
        return self.lmbks[:,:to_level+1].sum()

