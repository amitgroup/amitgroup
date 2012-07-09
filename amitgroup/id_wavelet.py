import amitgroup as ag
import amitgroup.math
import numpy as np
import pywt

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

def _gen_xs(shape):
    return np.mgrid[0:1.0:shape[0]*1j, 0:1.0:shape[1]*1j]

class IDWavelet(ag.ImageDeformation):
    def __init__(self, shape, rho=1.5):
        super(IDWavelet, self).__init__(shape)
        self.shape = shape
        self.rho = rho 
        self.invvar = 1.0/(shape[0]*shape[1])
        biggest = self.scriptNs[-1]        
        self.ushape = (2, self.levels+1, 3, biggest, biggest)
        self.u = np.zeros(self.ushape)
        self._init_lmbks()

    def _init_lmbks(self):
        self.lmbks = np.zeros(self.ushape)
        for i in range(self.levels+1):
            self.lmbks[:,i,:,:,:] = self.invvar * 2.0**(self.rho * i) 

    def _wl_name(self):
        return 'db2'

    def set_shape(self, shape):
        wl = self._wl_name()
        self.levels = len(pywt.wavedec(range(shape[0]), wl)) - 1
        self.scriptNs = map(len, pywt.wavedec(range(shape[0]), wl, level=self.levels))

    def _deformed_x(self, x0, x1):
        Ux0, Ux1 = self.deform_map(x0, x1)
        return x0+Ux0, x1+Ux1

    def get_x(self, shape):
        dx = 1./shape[0]
        dy = 1./shape[1]
        return np.mgrid[0:1.0-dx:shape[0]*1j, 0:1.0-dy:shape[1]*1j]
 
    def deform_map(self, x, y):
        """
        Creates a deformation array according the image deformation. 

        Parameters
        ----------
        x, y : ndarray
            Arrays of `x` and `y` values. Generate these by ``numpy.mgrid``. Array of shape ``(L, L)``.

        Returns
        -------
        Ux : ndarray
            Deformation along the `x` axis. Array of shape ``(L, L)``. 
        Uy : ndarray
            Same as above, along `y` axis. 
        """
        wl = pywt.Wavelet(self._wl_name())
        defx0 = pywt.waverec2(_array2pywt(self.u[0], self.scriptNs), wl) 
        defx1 = pywt.waverec2(_array2pywt(self.u[1], self.scriptNs), wl)

        # Interpolated defx at xs 
        Ux = ag.math.interp2d(x, y, defx0, dx=np.array([1.0/(defx0.shape[0]-1), 1.0/(defx0.shape[1]-1)]))
        Uy = ag.math.interp2d(x, y, defx1, dx=np.array([1.0/(defx1.shape[0]-1), 1.0/(defx1.shape[1]-1)]))
        return Ux, Uy 


    def deform(self, I):
        """
        Deform I according to Daubechies coefficients u.
        """
        im = np.zeros(I.shape)

        x0, x1 = self.get_x(I.shape) 
        z0, z1 = self._deformed_x(x0, x1)
        im = ag.math.interp2d(z0, z1, I)
        return im

    def logprior(self):
        return (self.lmbks * self.u**2).sum() / 2.0

    def reestimate(self, stepsize, delFz, Fz, I, level):
        terms = Fz - I
        wl = pywt.Wavelet(self._wl_name())
        vqks = np.array([
            _pywt2array(pywt.wavedec2(delFz[q] * terms, wl, level=self.levels), self.scriptNs, level) for q in range(2)
        ])

        self.u -= stepsize * (self.lmbks * self.u + vqks)

