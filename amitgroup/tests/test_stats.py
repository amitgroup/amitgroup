import amitgroup as ag
import numpy as np
import unittest
import os

def rel(x): return os.path.join(os.path.abspath(os.path.dirname(__file__)), x)

class TestStats(unittest.TestCase):
    def setUp(self):
        pass

    def test_bernoulli_mixture(self):
        nines = ag.io.load_example('mnist')
        bin_data = (nines > 0.5).astype(int)
        bm = ag.stats.BernoulliMixture(3, bin_data)
        bm.run_EM(1e-3, min_probability=0.05)
        
        # Save the data (when you're sure this test will succeed)
        #bm.save(rel('data/bernoulli_mixture_test.npz'), save_affinities=True)

        correct_data = np.load(rel('data/bernoulli_mixture_test.npz'))

        np.testing.assert_array_almost_equal(bm.templates, correct_data['templates'])
        np.testing.assert_array_almost_equal(bm.weights, correct_data['weights'])
        np.testing.assert_array_almost_equal(bm.affinities, correct_data['affinities'])

if __name__ == '__main__':
    unittest.main()

