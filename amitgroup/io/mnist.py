import os, struct
from array import array as pyarray 
from numpy import append, array, int8, uint8, zeros

def read(dataset = "training", path = ".", digits=None, asbytes=False):
    """
    Loads MNIST files into a 3D numpy arrays.

    Parameters
    ----------
    dataset : string
        Either "training" or "testing", depending on which dataset you want to load. 
    path : string
        Path of your MNIST datafiles. Can be downloaded here: http://yann.lecun.com/exdb/mnist/
    digits : array
        ...
    asbytes : bool
        If yes, returns data as np.uint8 as opposed to np.float64 range from 0.0 to 1.0.

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py 
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
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(images_fname, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    if digits:
        indices = [k for k in xrange(size) if lbl[k] in digits]
    else:
        indices = range(size)
    N = len(indices)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N), dtype=int8)
    for i in xrange(len(indices)):
        images[i] = array(img[ indices[i]*rows*cols : (indices[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[indices[i]]

    if asbytes:
        return images, labels
    else:
        return images/255.0, labels
