import amitgroup as ag
import numpy as np
import unittest
import os
import amitgroup.util.wavelet
import amitgroup as ag

def rel(x): return os.path.join(os.path.abspath(os.path.dirname(__file__)), x)

class TestWavelet(unittest.TestCase):
    def setUp(self):
        pass

    def _test_wavedec2d(self, wavelet, shape, levels):
        import pywt
        ll = int(np.log2(max(shape)))
        A = np.arange(np.prod(shape)).reshape(shape)

        coefs = pywt.wavedec2(A, wavelet, mode='per', level=ll)
        u_ref = ag.util.wavelet.pywt2array(coefs[:levels+1]) # including 0th level as one

        wavedec2 = ag.util.wavelet.daubechies2d_forward_factory(A.shape, levels=levels)
        u = wavedec2(A)

        np.testing.assert_array_almost_equal(u_ref, u)
        return u

    def test_wavedec2d_16(self):
        for i in range(1, 5):
            self._test_wavedec2d('db2', (16, 16), i)

    def test_wavedec2d_32(self):
        for i in range(1, 6):
            self._test_wavedec2d('db2', (32, 32), i)

    def test_wavedec2d_64(self):
        for i in range(1, 7):
            self._test_wavedec2d('db2', (64, 64), i)


    def _test_waverec2d(self, wavelet, shape, levels):
        A = np.arange(np.prod(shape)).reshape(shape)
        ll = int(np.log2(max(shape)))

        N = 1 << levels
        # This assumes wavedec2d is working
        u = self._test_wavedec2d(wavelet, shape, levels)  

        u_zeros = np.zeros(u.shape)
        u_zeros[:N,:N] = u[:N,:N]

        # Reconstruction
        waverec2 = ag.util.wavelet.daubechies2d_inverse_factory(A.shape, levels=levels)
    
        A_rec_ref = waverec2(u_zeros, levels)
        A_rec = waverec2(u, levels)

        

        if levels == ll:
            np.testing.assert_array_almost_equal(A, A_rec)
        else:
            # They should not be equal, since the image will have lost integrity
            assert not (A == A_rec).all()
            np.testing.assert_array_almost_equal(A_rec_ref, A_rec) 
        
    def test_waverec2d_16(self):
        for i in range(1, 5):
            self._test_waverec2d('db2', (16, 16), i)
        
    def test_waverec2d_32(self):
        for i in range(1, 6):
            self._test_waverec2d('db2', (32, 32), i)

    def test_waverec2d_64(self):
        for i in range(1, 7):
            self._test_waverec2d('db2', (64, 64), i)


if __name__ == '__main__':
    unittest.main()

