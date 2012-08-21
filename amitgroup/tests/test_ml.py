import amitgroup as ag
import numpy as np
import unittest
import os

def rel(x): return os.path.join(os.path.abspath(os.path.dirname(__file__)), x)

class TestML(unittest.TestCase):
    def setUp(self):
        self.I = np.zeros((8, 8))
        self.I[1:3,1:2] = 1.0

        self.F = np.zeros((8, 8))
        self.F[2:3,0:1] = 1.0

    def test_image_deformation(self):
        # Set tolerance pretty high, to make it fast
        imdef, info = ag.stats.image_deformation(self.F, self.I, last_level=2, penalty=1.0, rho=1.0, tol=1e-7)
        Fdef = imdef.deform(self.F)
    
        # Save the data (when you're sure this test will succeed)
        #np.save(rel('data/image_deformation_test.npy'), Fdef)

        Fdef_correct = np.load(rel('data/image_deformation_test.npy'))

        np.testing.assert_array_almost_equal(Fdef, Fdef_correct)

    def test_image_deformation_nonsquare(self):
        # Not complete
        if 0:
            F2 = F[::2] 
            imdef = ag.util.DisplacementFieldWavelet(F2.shape, 'db2')
            I2.randomize(2.0)
            I2 = imdef.deform(F2) 

        
        
    

if __name__ == '__main__':
    unittest.main()
