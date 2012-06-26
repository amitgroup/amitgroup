import os, struct
from array import array as pyarray 
from numpy import append, array, int8, uint8, zeros

def read(dataset = "training", path = ".", digits=None, asbytes=False):
    """
    Loads MNIST files into a 3D numpy arrays.

    Parameters
    ----------
    dataset : str 
        Either "training" or "testing", depending on which dataset you want to load. 
    path : str 
        Path to your MNIST datafiles. Can be downloaded here: http://yann.lecun.com/exdb/mnist/
    digits : list 
        Integer list of digits to load. The entire database is loaded if set to None. Default is None.
    asbytes : bool
        If True, returns data as ``numpy.uint8`` in [0, 255] as opposed to ``numpy.float64`` in [0.0, 1.0].

    Returns
    -------
    images : ndarray
        Image data of shape ``(N, rows, cols)``, where ``N`` is the number of images. 
    labels : ndarray
        Array of size ``N`` describing .

    Examples
    --------
    >>> from amitgroup.io.mnist import read 
    >>> images, labels = read('training', '/path/to/mnist')

    >>> sevens, _ = read('testing', '/path/to/mnist', [7])
    """

    # The files are assumed to have these names and should be found in 'path'
    files = {
        'training': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
        'testing': ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'),
    }

    try:
        images_fname = os.path.join(path, files[dataset][0])
        labels_fname = os.path.join(path, files[dataset][1])
    except KeyError:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(labels_fname, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    labels_raw = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(images_fname, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    images_raw = pyarray("B", fimg.read())
    fimg.close()

    if digits:
        indices = [k for k in xrange(size) if labels_raw[k] in digits]
    else:
        indices = range(size)
    N = len(indices)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N), dtype=int8)
    for i in xrange(len(indices)):
        images[i] = array(images_raw[ indices[i]*rows*cols : (indices[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = labels_raw[indices[i]]

    if asbytes:
        return images, labels
    else:
        return images/255.0, labels
