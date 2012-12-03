
import scipy.signal
import numpy as np
import amitgroup as ag

def _gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions. """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def _blur_and_shrink(im, n, ny=None):
    g = _gauss_kern(n, sizey=ny)
    improc = scipy.signal.convolve(im,g, mode='valid')
    return(improc)

def blur_image(im, n, ny=None, maintain_size=True):
    """ 
    Blurs the image by convolving with a gaussian kernel of typical
    size `n`. The optional keyword argument `ny` allows for a different
    size in the `y` direction.
    
    You can also use scipy.ndimage.filters.gaussian_filter.

    Parameters
    ----------
    im : ndarray
        2D array with an image.
    n : int
        Kernel size.
    ny : int
        Kernel size in y, if specified. 
    maintain_size : bool
        If True, the size of the image will be maintained. This is done by first padding the image with the edge values, before convolving.

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
    if maintain_size:
        if ny is None:
            ny = n 
        x, y = np.mgrid[-n:im.shape[0]+n, -ny:im.shape[1]+ny].astype(float)
        bigger = ag.util.interp2d(x, y, im.astype(float), startx=(0, 0), dx=(1, 1))
        return _blur_and_shrink(bigger, n, ny) 
    else:
        return _blur_and_shrink(im, n, ny)
