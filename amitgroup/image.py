from __future__ import division, print_function, absolute_import
import itertools as itr
import numpy as np


def resize_by_factor(im, factor):
    """
    Resizes the image according to a factor. The image is pre-filtered
    with a Gaussian and then resampled with bilinear interpolation.

    This function uses scikit-image and essentially combines its
    `pyramid_reduce` with `pyramid_expand` into one function.

    Returns the same object if factor is 1, not a copy.

    Parameters
    ----------
    im : ndarray, ndim=2 or 3
        Image. Either 2D or 3D with 3 or 4 channels.
    factor : float
        Resize factor, e.g. a factor of 0.5 will halve both sides.
    """
    from skimage.transform.pyramids import pyramid_reduce, pyramid_expand
    if factor < 1:
        return pyramid_reduce(im, downscale=1/factor)
    elif factor > 1:
        return pyramid_expand(im, upscale=factor)
    else:
        return im


def asgray(im):
    """
    Takes an image and returns its grayscale version by averaging the color
    channels.  if an alpha channel is present, it will simply be ignored. If a
    grayscale image is given, the original image is returned.

    Parameters
    ----------
    image : ndarray, ndim 2 or 3
        RGB or grayscale image.

    Returns
    -------
    gray_image : ndarray, ndim 2
        Grayscale version of image.
    """
    if im.ndim == 2:
        return im
    elif im.ndim == 3 and im.shape[2] in (3, 4):
        return im[..., :3].mean(axis=-1)
    else:
        raise ValueError('Invalid image format')


def crop(im, size):
    """
    Crops an image in the center.

    Parameters
    ----------
    size : tuple, (height, width)
        Finally size after cropping.
    """
    diff = [im.shape[index] - size[index] for index in (0, 1)]
    im2 = im[diff[0]//2:diff[0]//2 + size[0], diff[1]//2:diff[1]//2 + size[1]]
    return im2


def crop_to_bounding_box(im, bb):
    """
    Crops according to a bounding box.

    Parameters
    ----------
    bounding_box : tuple, (top, left, bottom, right)
        Crops inclusively for top/left and exclusively for bottom/right.
    """
    return im[bb[0]:bb[2], bb[1]:bb[3]]


def load(path, asfloat=True):
    """
    Loads an image from file.

    Parameters
    ----------
    path : str
        Path to image file.
    asfloat : bool
        Defaults to True, which means the image will be returned as a float
        with values between 0 and 1.
    """
    import skimage.io
    im = skimage.io.imread(path)
    if asfloat:
        return im.astype(np.float64) / 255
    else:
        return im


def save(path, im):
    """
    Saves an image to file.

    If the image is type float, it will assume to have values in [0, 1].

    Parameters
    ----------
    path : str
        Path to which the image will be saved.
    im : ndarray (image)
        Image.
    """
    from PIL import Image
    if im.dtype == np.uint8:
        pil_im = Image.fromarray(im)
    else:
        pil_im = Image.fromarray((im*255).astype(np.uint8))
    pil_im.save(path)


def integrate(ii, r0, c0, r1, c1):
    """
    Use an integral image to integrate over a given window.

    Parameters
    ----------
    ii : ndarray
        Integral image.
    r0, c0 : int
        Top-left corner of block to be summed.
    r1, c1 : int
        Bottom-right corner of block to be summed.

    Returns
    -------
    S : int
        Integral (sum) over the given window.

    """
    # This line is modified
    S = np.zeros(ii.shape[-1])

    S += ii[r1, c1]

    if (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S += ii[r0 - 1, c0 - 1]

    if (r0 - 1 >= 0):
        S -= ii[r0 - 1, c1]

    if (c0 - 1 >= 0):
        S -= ii[r1, c0 - 1]

    return S


def offset(img, offset, fill_value=0):
    """
    Moves the contents of image without changing the image size. The missing
    values are given a specified fill value.

    Parameters
    ----------
    img : array
        Image.
    offset : (vertical_offset, horizontal_offset)
        Tuple of length 2, specifying the offset along the two axes.
    fill_value : dtype of img
        Fill value. Defaults to 0.
    """

    sh = img.shape
    if sh == (0, 0):
        return img
    else:
        x = np.empty(sh)
        x[:] = fill_value
        x[max(offset[0], 0):min(sh[0]+offset[0], sh[0]),
          max(offset[1], 0):min(sh[1]+offset[1], sh[1])] = \
            img[max(-offset[0], 0):min(sh[0]-offset[0], sh[0]),
                max(-offset[1], 0):min(sh[1]-offset[1], sh[1])]
        return x


def bounding_box(alpha, threshold=0.1):
    """
    Returns a bounding box of the support.

    Parameters
    ----------
    alpha : ndarray, ndim=2
        Any one-channel image where the background has zero or low intensity.
    threshold : float
        The threshold that divides background from foreground.

    Returns
    -------
    bounding_box : (top, left, bottom, right)
        The bounding box describing the smallest rectangle containing the
        foreground object, as defined by the threshold.
    """
    assert alpha.ndim == 2

    # Take the bounding box of the support, with a certain threshold.
    supp_axs = [alpha.max(axis=1-i) for i in range(2)]

    # Check first and last value of that threshold
    bb = [np.where(supp_axs[i] > threshold)[0][[0, -1]] for i in range(2)]

    return (bb[0][0], bb[1][0], bb[0][1], bb[1][1])


def bounding_box_as_binary_map(alpha, threshold=0.1):
    """
    Similar to `bounding_box`, except returns the bounding box as a
    binary map the same size as the input.

    Same parameters as `bounding_box`.

    Returns
    -------
    binary_map : ndarray, ndim=2, dtype=np.bool_
        Binary map with True if object and False if background.
    """

    bb = bounding_box(alpha)
    x = np.zeros(alpha.shape, dtype=np.bool_)
    x[bb[0]:bb[2], bb[1]:bb[3]] = 1
    return x


def extract_patches(images, patch_shape, samples_per_image=40, seed=0,
                    cycle=True):
    """
    Takes a set of images and yields randomly chosen patches of specified size.

    Parameters
    ----------
    images : iterable
        The images have to be iterable, and each element must be a Numpy array
        with at least two spatial 2 dimensions as the first and second axis.
    patch_shape : tuple, length 2
        The spatial shape of the patches that should be extracted. If the
        images have further dimensions beyond the spatial, the patches will
        copy these too.
    samples_per_image : int
        Samples to extract before moving on to the next image.
    seed : int
        Seed with which to select the patches.
    cycle : bool
        If True, then the function will produce patches indefinitely, by going
        back to the first image when all are done. If False, the iteration will
        stop when there are no more images.

    Returns
    -------
    patch_generator
        This function returns a generator that will produce patches.

    Examples
    --------
    >>> import amitgroup as ag
    >>> import matplotlib.pylab as plt
    >>> import itertools
    >>> images = ag.io.load_example('mnist')

    Now, let us say we want to exact patches from the these, where each patch
    has at least some activity.

    >>> gen = ag.image.extract_patches(images, (5, 5))
    >>> gen = (x for x in gen if x.mean() > 0.1)
    >>> patches = np.array(list(itertools.islice(gen, 25)))
    >>> patches.shape
    (25, 5, 5)
    >>> ag.plot.images(patches)
    >>> plt.show()

    """
    rs = np.random.RandomState(seed)
    for Xi in itr.cycle(images):
        # How many patches could we extract?
        w, h = [Xi.shape[i]-patch_shape[i] for i in range(2)]

        assert w > 0 and h > 0

        # Maybe shuffle an iterator of the indices?
        indices = np.asarray(list(itr.product(range(w), range(h))))
        rs.shuffle(indices)
        for x, y in indices[:samples_per_image]:
            yield Xi[x:x+patch_shape[0], y:y+patch_shape[1]]
