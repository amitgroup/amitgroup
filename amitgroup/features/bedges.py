
from __future__ import absolute_import
import numpy as np
import scipy.signal
import amitgroup as ag
from amitgroup.features.features import array_bedges

# Builds a kernel along the edge direction
def _along_kernel(direction, radius):
    d = direction%4
    kern = None
    if d == 0: # S/N
        kern = np.zeros((radius*2+1,)*2, dtype=np.uint8)
        kern[radius,:] = 1
    elif d == 2: # E/W
        kern = np.zeros((radius*2+1,)*2, dtype=np.uint8)
        kern[:,radius] = 1
    elif d == 1: # SE/NW
        kern = np.eye(radius*2+1, dtype=np.uint8)[::-1]
    elif d == 3: # NE/SW
        kern = np.eye(radius*2+1, dtype=np.uint8)
            
    return kern

def bedges(images, k=6, inflate='box', radius=1, minimum_contrast=0.0, contrast_insensitive=False, firstaxis=False):
    """
    Extracts binary edge features for each pixel according to [1].

    The function returns 8 different binary features, representing directed edges. Let us define a south-going edge as when it starts at high intensity and drops when going south (this would make south edges the lower edge of an object, if background is low intensity and the object is high intensity). By this defintion, the order of the returned edges is S, SE, E, NE, N, NW, W, SW.

    Parameters
    ----------
    images : ndarray
        Input an image of shape ``(rows, cols)`` or a list of images as an array of shape ``(N, rows, cols)``, where ``N`` is the number of images, and ``rows`` and ``cols`` the size of each image.
    k : int
        There are 6 contrast differences that are checked. The value `k` specifies how many of them must be fulfilled for an edge to be present. The default is all of them (`k` = 6) and gives more conservative edges.
    inflate : 'box', 'perpendicular', None 
        If set to `'box'` and `radius` is set to 1, then an edge will appear if any of the 8 neighboring pixels detected an edge. This is equivalent to inflating the edges area with 1 pixel. The size of the box is dictated by `radius`. 
        If `'perpendicular'`, then the features will be extended by `radius` perpendicular to the direction of the edge feature (i.e. along the edge).
    radius : int
        Controls the extent of the inflation, see above.
    minimum_contrast : double
        Requires the gradient to have an absolute value greater than this, for an edge to be detected. Set to a non-zero value to reduce edges firing in low contrast areas.
    contrast_insensitive : bool
        If this is set to True, then the direction of the gradient does not matter and only 4 edge features will be returned.
    firstaxis: bool
         If True, the images will be returned with the features on the first axis as ``(A, rows, cols)`` instead of ``(rows, cols, A)``, where `A` is either 4 or 8. If mutliple input entries, then the output will be ``(N, A, rows, cols)``.
    
    Returns
    -------
    edges : ndarray
        An array of shape ``(rows, cols, A)`` if entered as a single image, or ``(N, rows, cols, A)`` of multiple. Each pixel in the original image becomes a binary vector of size 8, one bit for each cardinal and diagonal direction. 

    References
    ----------
    [1] Y. Amit : 2D Object Detection and Recognition: Models, Algorithms and Networks. Chapter 5.4.
    """
    single = len(images.shape) == 2
    if single:
        features = array_bedges(np.array([images]), k, minimum_contrast, contrast_insensitive)
    else:
        features = array_bedges(images, k, minimum_contrast, contrast_insensitive) 

    if inflate is True or inflate == 'box':
        features = ag.util.inflate2d(features, np.ones((1+radius*2, 1+radius*2)))
    elif inflate == 'along':
        # Propagate the feature along the edge 
        for j in xrange(8):
            kernel = _along_kernel(j, radius)
            features[:,j] = ag.util.inflate2d(features[:,j], kernel)

    if not firstaxis:
        features = np.rollaxis(features, axis=1, start=features.ndim)
            
    if single:
        features = features[0]

    return features

def bedges_from_image(im, k=6, inflate='box', radius=1, minimum_contrast=0.0, contrast_insensitive=False, firstaxis=False, return_original=False):
    """
    This wrapper for :func:`bedges`, will take an image file, load it and compute binary edges for each color channel separately, and then finally OR the result.

    Parameters
    ----------
    im : str / ndarray
        This can be either a string with a filename to an image file, or an ndarray of three dimensions, where the third dimension is the color channels.

    Returns
    -------
    edges : ndarray
        An array of shape ``(8, rows, cols)`` if entered as a single image, or ``(N, 8, rows, cols)`` of multiple. Each pixel in the original image becomes a binary vector of size 8, one bit for each cardinal and diagonal direction. 
    return_original : bool
        If True, then the original image is returned as well as the edges.
    image : ndarray
        An array of shape ``(rows, cols, D)``, where `D` is the number of color channels, probably 3 or 4. This is only returned if `return_original` is set to True.

    The rest of the argument are the same as :func:`bedges`.
    """
    if isinstance(im, str) or isinstance(im, file):
        from PIL import Image
        im = np.array(Image.open(im))
        # TODO: This needs more work. We should probably make bedges work with any type
        # and then just leave it at that.
        if im.dtype == np.uint8:
            im = im.astype(np.float64)/255.0

    # Run bedges on each channel, and then OR it. 
    dimensions = im.shape[-1]
    
    # This will use all color channels, including alpha, if there is one
    edges = [bedges(im[...,i], k=k, inflate=inflate, radius=radius, minimum_contrast=minimum_contrast, contrast_insensitive=contrast_insensitive, firstaxis=firstaxis) for i in xrange(dimensions)]

    final = reduce(np.bitwise_or, edges)

    if return_original:
        return final, im
    else:
        return final
    
