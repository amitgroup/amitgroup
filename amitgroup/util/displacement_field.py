import numpy as np

class DisplacementField(object):
    """
    Displacement field. Abstract base class for representing a deformation of a 2D mesh grid. 
    """
    def __init__(self, shape):
        self.shape = shape
        self.prepare_shape()

    def prepare_shape(self, shape):
        """
        Prepare shape. Subclasses can for instance prepare the 
        maximum number of coefficients appropriate for the shape.
        """
        pass

    def deform_map(self, x, y):
        """
        Creates a deformation array according the image deformation. 

        Parameters
        ----------
        x, y : ndarray
            Arrays of `x` and `y` values. Generate these by ``numpy.mgrid``. Array of shape ``(L, L)``.

        Returns
        -------
        Ux : ndarray
            Deformation along the `x` axis. Array of shape ``(L, L)``. 
        Uy : ndarray
            Same as above, along `y` axis. 
        """
        raise NotImplemented("Can't use DisplacementField directly")
        
    def deform(self, F):
        """
        Deforms the image F according to this displacement field.

        Parameters
        ----------
        F : ndarray
            2D array of data.
        
        Returns
        -------
        Fdef : ndarray
            2D array of the same size as `F`, representing a deformed version of `F`. 
        """
        raise NotImplemented("Can't use DisplacementField directly")

    @classmethod
    def get_x(cls, shape):
        """
        Returns a mesh of `x` and `y` values appropriate to use with this displacement field.

        Parameters
        ----------
        shape : tuple
            Tuple of length 2 specifying the size of the mesh. 
        
        Returns
        -------
        x : ndarray
            Contains `x` values. Array of shape specified above.
        y : ndarray
            Same as above, for `y` values. 

        Examples
        --------
        >>> x, y = DisplacementField.get_x((4, 4))
        >>> x
        array([[ 0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.25,  0.25,  0.25,  0.25],
               [ 0.5 ,  0.5 ,  0.5 ,  0.5 ],
               [ 0.75,  0.75,  0.75,  0.75]])

        """
        dx = 1./shape[0]
        dy = 1./shape[1]
        return np.mgrid[0:1.0-dx:shape[0]*1j, 0:1.0-dy:shape[1]*1j]
 
    def __repr__(self):
        return "{0}(shape={1})".format(self.__class__.__name__, self.shape)
