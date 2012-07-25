
from __future__ import absolute_import
from .features import array_bedges
import numpy as np

def bedges(images, inflate=True):
    """
    Extracts binary edge features for each pixel according to [1].

    Parameters
    ----------
    images : ndarray
        Input an image of shape ``(rows, cols)`` or a list of images as an array of shape ``(N, rows, cols)``, where ``N`` is the number of images, and ``rows`` and ``cols`` the size of each image.
    inflate : bool
        If True, then an edge will appear if any of the 8 neighboring pixels detected an edge. This is equivalent to inflating the edges area with 1 pixel. This adds robustness to your features.

    Returns
    -------
    edges : ndarray
        An array of shape ``(rows, cols, 8)`` if entered as a single image, or ``(N, rows, cols, 8)`` of multiple. Each pixel in the original image becomes a binary vector of size 8, one bit for each cardinal and diagonal direction. 

    References
    ----------
    [1] Y. Amit : 2D Object Detection and Recognition: Models, Algorithms and Networks. Chapter 5.4.
    """
    single = len(images.shape) == 2
    if single:
        features = array_bedges(np.array([images]))[0]
    else:
        features = array_bedges(images) 

    if inflate:
        # Will not inflate the one-pixel border aroudn the image.
        # Just try not to keep salient features that close to the edge!
        features[1:-1,1:-1,:] = features[1:-1,1:-1,:] | \
            features[:-2,1:-1,:] | features[2:,1:-1,:] | \
            features[1:-1,:-2,:] | features[1:-1,2:,:] | \
            features[:-2,:-2,:] | features[2:,2:,:] | \
            features[:-2,2:,:] | features[2:,:-2,:]
            
    return features 


