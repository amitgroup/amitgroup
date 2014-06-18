from __future__ import division, print_function, absolute_import
import numpy as np
import amitgroup as ag

# If skimage is available, the image returned will be wrapped
# in the Image class. This is nice since it will be automatically
# displayed in an IPython notebook.
try:
    from skimage.io import Image
except ImportError:
    def Image(x): x

class ImageGrid:
    """
    An image grid used for combining equally-sized intensity images into a
    single larger image.

    Parameters
    ----------
    rows, cols : int
        Rows and columns of the grid.
    shape : tuple, (height, width)
        Shape of each patch in the grid.
    border_color : array_like, length 3 or int/float
        Specify the color of the border between each entry as RGB values
        between 0 and 1. If a scalar is provided, it is interpreted as a
        grayscale intensity.
    border_width : int
        Pixel width of border.

    Examples
    --------

    >>> import amitgroup as ag
    >>> import numpy as np
    >>> import matplotlib.pylab as plt
    >>> from matplotlib.pylab import cm
    >>> rs = np.random.RandomState(0)

    Let's generate a set of 100 8x8 image patches. 

    >>> shape = (100, 8, 8)
    >>> images = np.arange(np.prod(shape)).reshape(shape)
    >>> images += rs.uniform(0, np.prod(shape), size=shape)

    The most convenient way of using `ImageGrid` is through
    `ImageGrid.fromarray`:

    >>> grid = ag.plot.ImageGrid.fromarray(images, cmap=cm.hsv)
    >>> img = grid.scaled_image(scale=5)
    >>> plt.imshow(img)
    >>> plt.show()

    If you are working in an IPython notebook, you can display
    ``img`` simply by adding it to the end of a cell.
    """
    def __init__(self, rows, cols, shape, border_color=None, border_width=1):
        if border_color is None:
            self._border_color = np.array([0.5, 0.5, 0.5])
        elif isinstance(border_color, (int, float)): 
            self._border_color = np.array([border_color]*3)
        else:
            self._border_color = np.array(border_color)
        self._rows = rows
        self._cols = cols
        self._shape = shape 
        self._border = border_width

        self._fullsize = (self._border + (shape[0] + self._border) * self._rows,
                          self._border + (shape[1] + self._border) * self._cols)

        self._data = np.ones(self._fullsize + (3,), dtype=np.float64)

    @classmethod
    def fromarray(cls, images, rows=None, cols=None, border_color=None,
            border_width=1, cmap=None, vmin=None, vmax=None,
            global_bounds=True):
        """
        Constructs an image grid from an array of images. The images will be
        layed out row by row starting from the top (as English is written).

        Parameters
        ----------
        images : ndarray, ndim=3
            The first dimension should index the image and the rest are spatial
            dimensions.
        rows, cols : int or None
            The number of rows and columns for the grid. If both are None, the
            minimal square grid that holds all images will be used. If one is
            specified, the other will adapt to hold all images. If both are
            specified, then it possible that the grid will be vacuous or that
            some images will be omitted.
        border_color, border_width :
            See `ImageGrid`.
        cmap, vmin, vmax :
            See `ImageGrid.set_image`.
        global_bounds : bool
            If this is set to True and either `vmin` or `vmax` is not specified, it
            will infer it globally for the data. If it is set to False, it does
            it per image, which would be the equivalent of calling `set_image`
            manually with `vmin` and `vmax` set as such.

        """
        assert images.ndim == 3
        N = images.shape[0]

        if rows is None and cols is None:
            rows = cols = int(np.ceil(np.sqrt(N)))
        elif rows is None:
            rows = int(np.ceil(N / cols))
        elif cols is None:
            cols = int(np.ceil(N / rows))

        grid = cls(rows, cols, images.shape[1:3],
                   border_color=border_color, border_width=border_width)

        if global_bounds:
            if vmin is None:
                vmin = images.min()
            if vmax is None:
                vmax = images.max()

        # Populate with images
        for i in range(min(N, rows * cols)):
            grid.set_image(images[i], i // cols, i % cols,
                           cmap=cmap, vmin=vmin, vmax=vmax)

        return grid

    @property
    def image(self):
        """
        Returns the image as a skimage.io.Image class.
        """
        return Image(self._data)

    def set_image(self, image, row, col, cmap=None, vmin=None, vmax=None):
        """
        Sets the data for a single window.

        Parameters
        ----------
        image : ndarray, ndim=2
            The shape should be the same as the `shape` specified when
            constructing the image grid.
        row, col : int
            The zero-index of the row and column to set.
        cmap : cmap (from matplotlib.pylab.cm)
            The color palette to use. Default is grayscale.
        vmin, vmax : numerical or None
            Defines the range of the color palette. None, which is default,
            takes the range of the data.
        """
        import matplotlib as mpl
        import matplotlib.pylab as plt
        from amitgroup.plot.resample import resample_and_arrange_image

        if cmap is None:
            cmap = plt.cm.gray
        if vmin is None:
            vmin = image.min()
        if vmax is None:
            vmax = image.max()

        if vmin == vmax:
            diff = 1
        else:
            diff = vmax - vmin

        image_indices = np.clip((image - vmin) / diff, 0, 1) * 255
        image_indices = image_indices.astype(np.uint8)

        lut = mpl.colors.makeMappingArray(256, cmap)
        rgb = resample_and_arrange_image(image_indices, self._shape, lut)

        x0 = row * (self._shape[0] + self._border)
        x1 = (row + 1) * (self._shape[0] + self._border) + self._border
        y0 = col * (self._shape[1] + self._border)
        y1 = (col + 1) * (self._shape[1] + self._border) + self._border

        self._data[x0:x1, y0:y1] = self._border_color

        anchor = (self._border + row * (self._shape[0] + self._border),
                  self._border + col * (self._shape[1] + self._border))

        self._data[anchor[0]:anchor[0] + rgb.shape[0],
                   anchor[1]:anchor[1] + rgb.shape[1]] = rgb

    def scaled_image(self, scale=1):
        """
        Returns a nearest-neighbor upscaled scaled version of the image.

        Parameters
        ----------
        scale : int
            Upscaling using nearest neighbor, e.g. a scale of 5 will make each
            pixel a 5x5 rectangle in the output.

        Returns
        -------
        scaled_image : skimage.io.Image, (height, width, 3)
            Returns a scaled up RGB image. If you do not have scikit-image, it
            will be returned as a regular Numpy array. The benefit of wrapping
            it in `Image`, is so that it will be automatically displayed in
            IPython notebook, without having to issue any drawing calls.
        """

        if scale == 1:
            return self._data
        else:
            from skimage.transform import resize
            data = resize(self._data, tuple([self._data.shape[i] * scale
                                             for i in range(2)]), order=0)
            # Make sure the borders stay the same color
            data[:scale * self._border] = self._border_color
            data[-scale * self._border:] = self._border_color
            data[:, :scale * self._border] = self._border_color
            data[:, -scale * self._border:] = self._border_color
            return Image(data)

    def save(self, path, scale=1):
        """
        Save the image to file.

        Parameters
        ----------
        path : str
            Output path.
        scale : int
            Upscaling using nearest neighbor, e.g. a scale of 5 will make each
            pixel a 5x5 rectangle in the output.
        """

        data = self.scaled_image(scale)
        ag.image.save(path, data)

    def __repr__(self):
        return 'ImageGrid(rows={rows}, cols={cols}, shape={shape})'.format(
                    rows=self._rows,
                    cols=self._cols,
                    shape=self._shape)
