import amitgroup as ag
import amitgroup.io
import unittest

class TestIO(unittest.TestCase):
    def setUp(self):
        pass

    def test_example(self):
        data = ag.io.load_example('two-faces')
        self.assertEqual(data.shape, (2, 32, 32))

        data2 = ag.io.load_example('mnist')
        self.assertEqual(data2.shape, (10, 28, 28))
    
    @unittest.skip("Mnist is optional")
    def test_mnist(self):
        try:
            digits, _ = ag.io.load_mnist('training')
        except IOError:
            assert 0, "No MNIST data files" 


if __name__ == '__main__':
    unittest.main()
