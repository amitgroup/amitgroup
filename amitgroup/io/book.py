# Reading files from Yali Amit's book
# Author: Gustav Larsson
import struct
from array import array as python_array 
import numpy as np

__all__ = ['load_image', 'load_all_images']

def _get_buf_and_length(filename):
    f = open(filename, 'rb')
    buf = f.read()
    N, = struct.unpack('>I', buf[:4]) 
    return N, buf

def _unpack_image(buf, index, asbytes):
    # Get offset
    offset, = struct.unpack('>I', buf[4+index*4:4+index*4+4])

    # Get width and height
    width, height = struct.unpack('>II', buf[4+offset:4+offset+8])

    # Finally get the image data put into numpy array
    image = np.array(python_array('B', buf[4+offset+8:4+offset+8+width*height])).reshape((width, height))

    if asbytes:
        return image 
    else:
        return image/255.0

def load_image(filename, index, asbytes=False):
    """
    Load one image from Yali Amit's book, specifically the FACES data.

    Parameters
    ----------
    filename: str 
        Filename with absolute or relative path.
    index: int
        Which image in the file to read.
    asbytes : bool
        If True, returns data as ``numpy.uint8`` in [0, 255] as opposed to ``numpy.float64`` in [0.0, 1.0].

    Returns
    -------
    image:
        Image data of shape `(rows, cols)`.
    """

    # TODO: This shouldn't have to load the entire image buffer!
    
    N, buf = _get_buf_and_length(filename)
    if not (0 <= index < N):
        raise TypeError("Invalid index")

    return _unpack_image(buf, index, asbytes)

def load_all_images(filename, asbytes=False):
    """
    Load images from Yali Amit's book, specifically the FACES data.

    Parameters
    ----------
    filename: str 
        Filename with absolute or relative path.
    asbytes : bool
        If True, returns data as ``numpy.uint8`` in [0, 255] as opposed to ``numpy.float64`` in [0.0, 1.0].

    Returns
    -------
    image:
        Image data of shape ``(N, rows, cols)``, where ``N`` is the number of images. 
    """

    N, buf = _get_buf_and_length(filename)
    images = []
    for i in xrange(N):
        images.append(_unpack_image(buf, i, asbytes))
    
    return np.array(images) 
