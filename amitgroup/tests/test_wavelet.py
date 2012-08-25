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

    def _test_wavedec2(self, wavelet, shape, levels):
        import pywt
        ll = int(np.log2(max(shape)))
        A = np.arange(np.prod(shape)).reshape(shape)

        coefs = pywt.wavedec2(A, wavelet, mode='per', level=ll)
        u_ref = ag.util.wavelet.pywt2array(coefs[:levels+1]) # including 0th level as one

        wavedec2 = ag.util.wavelet.wavedec2_factory(A.shape, wavelet=wavelet, levels=levels)
        u = wavedec2(A)

        np.testing.assert_array_almost_equal(u_ref, u)
        return u

    def test_wavedec2_16(self):
        for i in range(1, 5):
            self._test_wavedec2('db2', (16, 16), i)

    def test_wavedec2_32(self):
        for i in range(1, 6):
            self._test_wavedec2('db2', (32, 32), i)

    def test_wavedec2_64(self):
        for i in range(1, 7):
            self._test_wavedec2('db2', (64, 64), i)


    def _test_waverec2(self, wavelet, shape, levels):
        A = np.arange(np.prod(shape)).reshape(shape)
        ll = int(np.log2(max(shape)))

        N = 1 << levels
        # This assumes wavedec2 is working
        u = self._test_wavedec2(wavelet, shape, levels)  

        u_zeros = np.zeros(u.shape)
        u_zeros[:N,:N] = u[:N,:N]

        # Reconstruction
        waverec2 = ag.util.wavelet.waverec2_factory(A.shape, wavelet=wavelet, levels=levels)
    
        A_rec_ref = waverec2(u_zeros)
        A_rec = waverec2(u)

        

        if levels == ll:
            np.testing.assert_array_almost_equal(A, A_rec)
        else:
            # They should not be equal, since the image will have lost integrity
            assert not (A == A_rec).all()
            np.testing.assert_array_almost_equal(A_rec_ref, A_rec) 
        
    def test_waverec2_16(self):
        for i in range(1, 5):
            self._test_waverec2('db2', (16, 16), i)
        
    def test_waverec2_32(self):
        for i in range(1, 6):
            self._test_waverec2('db2', (32, 32), i)

    def test_waverec2_64(self):
        for i in range(1, 7):
            self._test_waverec2('db2', (64, 64), i)

    # Non-square
    @unittest.skip("Not implemented yet.")
    def test_wavedec2_32_16(self):
        for i in range(1, 5):
            self._test_wavedec2('db2', (32, 16), i)

    @unittest.skip("Not implemented yet.")
    def test_wavedec2_16_32(self):
        for i in range(1, 5):
            self._test_wavedec2('db2', (16, 32), i)

    # Test all wavelet types.
    def test_wavedec2_all_daubechies(self):
        for i in range(1, 21):
            self._test_wavedec2('db{0}'.format(i), (8, 8), i)

    def test_waverec2_all_daubechies(self):
        for i in range(1, 21):
            self._test_waverec2('db{0}'.format(i), (8, 8), i)

if __name__ == '__main__':
    unittest.main()

