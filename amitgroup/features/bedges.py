
from __future__ import absolute_import
from .features import array_bedges
import numpy as np

def bedges(images, k=6, inflate=True, lastaxis=False):
    """
    Extracts binary edge features for each pixel according to [1].

    Parameters
    ----------
    images : ndarray
        Input an image of shape ``(rows, cols)`` or a list of images as an array of shape ``(N, rows, cols)``, where ``N`` is the number of images, and ``rows`` and ``cols`` the size of each image.
    k : int
        There are 6 contrast differences that are checked. The value `k` specifies how many of them must be fulfilled for an edge to be present. The default is all of them (`k` = 6) and gives more conservative edges.
    inflate : bool
        If True, then an edge will appear if any of the 8 neighboring pixels detected an edge. This is equivalent to inflating the edges area with 1 pixel. This adds robustness to your features.
    lastaxis: bool
        If True, the images will be returned with the features on the last axis as ``(rows, cols, 8)`` instead of ``(8, rows, cols)``. 
    
    Returns
    -------
    edges : ndarray
        An array of shape ``(8, rows, cols)`` if entered as a single image, or ``(N, 8, rows, cols)`` of multiple. Each pixel in the original image becomes a binary vector of size 8, one bit for each cardinal and diagonal direction. 

    References
    ----------
    [1] Y. Amit : 2D Object Detection and Recognition: Models, Algorithms and Networks. Chapter 5.4.
    """
    single = len(images.shape) == 2
    if single:
        features = array_bedges(np.array([images]), k)
    else:
        features = array_bedges(images, k) 

    if inflate:
        # Will not touch the one-pixel border around the image.
        # Just try not to keep salient features that close to the edge!
        for feature in features:
            feature[1:-1,1:-1] = feature[1:-1,1:-1] | \
                feature[:-2,1:-1] | feature[2:,1:-1] | \
                feature[1:-1,:-2] | feature[1:-1,2:] | \
                feature[:-2,:-2] | feature[2:,2:] | \
                feature[:-2,2:] | feature[2:,:-2]
            
    if single:
        features = features[0]
    if not lastaxis:
        features = np.rollaxis(features, axis=-1, start=features.ndim-3)

    return features

