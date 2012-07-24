import amitgroup as ag
import amitgroup.ml
import numpy as np
import unittest

class TestML(unittest.TestCase):
    def setUp(self):
        self.I = np.zeros((8, 8))
        self.I[1:3,1:2] = 1.0

        self.F = np.zeros((8, 8))
        self.F[2:3,0:1] = 1.0

    def test_imagedef(self):
        imdef, info = ag.ml.imagedef(self.F, self.I)
        Fdef = imdef.deform(self.F)
        Fdef_correct = np.zeros((8, 8))
        Fdef_correct[1,0] = 0.01197697
        Fdef_correct[1,1] = 0.36150678
        Fdef_correct[2,0] = 0.01559739
        Fdef_correct[2,1] = 0.83793517
        Fdef_correct[3,1] = 0.00264384

        np.testing.assert_array_almost_equal(Fdef, Fdef_correct)

if __name__ == '__main__':
    unittest.main()
