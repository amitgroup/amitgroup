
from scipy.signal import convolve
import numpy as np

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions. """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None) :
    """ 
    Blurs the image by convolving with a gaussian kernel of typical
    size `n`. The optional keyword argument `ny` allows for a different
    size in the `y` direction.

    Examples
    --------
    >>> import amitgroup as ag
    >>> import matplotlib.pylab as plt

    Blur an image of a face:

    >>> face = ag.io.load_example('faces')[0]
    >>> face2 = ag.util.blur_image(face, 5)
    >>> ag.plot.images([face, face2]) 
    >>> plt.show()
    """
    g = gauss_kern(n, sizey=ny)
    improc = convolve(im,g, mode='valid')
    return(improc)
