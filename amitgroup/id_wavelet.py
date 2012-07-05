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
    def __init__(self):
        super(IDWavelet, self).__init__()
        self.u = None 
        self.scriptNs = None
        self.wl_name = 'db2'

    def _deformed_x(self, x0, x1):
        Ux0, Ux1 = deform_map(x0, x1)
        return x0+Ux0, x1+Ux1
 
    def deform_map(self, x, y):
        defx0 = pywt.waverec2(_array2pywt(self.u[0], self.scriptNs), self.wl_name) 
        defx1 = pywt.waverec2(_array2pywt(self.u[1], self.scriptNs), self.wl_name)

        # Interpolated defx at xs 
        Ux = ag.math.interp2d(x, y, defx0, dx=np.array([1.0/(defx0.shape[0]-1), 1.0/(defx0.shape[1]-1)]))
        Uy = ag.math.interp2d(x, y, defx1, dx=np.array([1.0/(defx1.shape[0]-1), 1.0/(defx1.shape[1]-1)]))
        return Ux, Uy 


    def deform(self, I):
        """
        Deform I according to Daubechies coefficients u.
        """
        im = np.zeros(I.shape)

        x0, x1 = _gen_xs(im.shape)

        z0, z1 = self._deformed_x(x0, x1)
        im = ag.math.interp2d(z0, z1, I)
        return im

