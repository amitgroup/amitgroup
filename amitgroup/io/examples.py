
import amitgroup as ag
import amitgroup.io
import numpy as np
import os

def datapath(name): 
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data', name) 

def load_example(name):
    """
    Loads example data.

    The `amitgroup` package comes with some subsets of data sets, to use in examples and for quick testing.
    
    Parameters
    ----------
    name : str
        - ``"faces"``: loads ``N`` faces into an array of shape ``(N, rows, cols)``.
        - ``"faces2"``: loads 2 faces into an array of shape ``(2, rows, cols)``.
        - ``"mnist"``: loads 10 MNIST nines into an array of shape ``(10, rows, cols)``.

    Examples
    --------
    >>> import amitgroup as ag
    >>> import matplotlib.pylab as plt

    Load faces:

    >>> faces = ag.io.load_example("faces")
    >>> for i in range(6):
    >>>     plt.subplot(231+i)
    >>>     plt.imshow(faces[i], cmap=plt.cm.gray, interpolation='nearest')
    >>> plt.show()

    Load MNIST:

    >>> digits = ag.io.load_example("mnist")
    >>> for i in range(6):
    >>>     plt.subplot(231+i)
    >>>     plt.imshow(digits[i], cmap=plt.cm.gray_r, interpolation='nearest')
    >>> plt.show()

    """
    if name == 'faces':
        return ag.io.load_all_images(datapath('../data/Images_0'))
    if name == 'faces2':
        data = np.load(datapath('../data/twoface.npz'))
        return np.array([data['im1'], data['im2']])
    if name == 'mnist':
        return np.load(datapath('../data/nines.npz'))['images']
    else:
        raise ValueError("Example data does not exist")
