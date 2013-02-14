import numpy as np

class Saveable(object):
    """
    Key-value coding interface for classes. Generally, this is an interface that make it possible to
    access instance members through keys (strings), instead of through named variables. What this
    interface enables, is to save and load an instance of the class to file. This is done by encoding
    it into a dictionary, or decoding it from a dictionary. The dictionary is then saved/loaded using
    numpy's ``npy`` files. 
    """
    @classmethod
    def load(cls, path):
        """
        Loads an instance of the class from a file.

        Parameters
        ----------
        path : str
            Path to an ``npy`` file. 

        Examples
        --------
        This is an abstract data type, but let us say that ``Foo`` inherits from ``Saveable``. To construct
        an object of this class from a file, we do:

        >>> foo = Foo.load(path) #doctest: +SKIP
        """
        if path is None:
            return cls.load_from_dict({})
        else:
            d = np.load(path).flat[0]
            return cls.load_from_dict(d)
        
    def save(self, path):
        """
        Saves an instance of the class to a numpy ``npy`` file.

        Parameters
        ----------
        path : str
            Output path. If no file extension is specified, it will be saved as ``.npy``.
        """
        np.save(path, self.save_to_dict())

    @classmethod
    def load_from_dict(cls, d):
        """
        Overload this function in your subclass. It takes a dictionary and should return a constructed object.

        When overloading, you have to decorate this function with ``@classmethod``.

        Parameters
        ----------
        d : dict
            Dictionary representation of an instance of your class.

        Returns
        -------
        obj : object
            Returns an object that has been constructed based on the dictionary.
        """
        raise NotImplementedError("Must override load_from_dict for Saveable interface")

    def save_to_dict(self):
        """
        Overload this function in your subclass. It should return a dictionary representation of
        the current instance. 

        If you member variables that are objects, it is best to convert them to dictionaries
        before they are entered into your dictionary hierarchy.

        Returns
        -------
        d : dict 
            Returns a dictionary representation of the current instance. 
        """
        raise NotImplementedError("Must override save_to_dict for Saveable interface")

