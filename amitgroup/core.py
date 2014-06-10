from __future__ import division, print_function, absolute_import
_is_verbose = False
_is_silent = False

import warnings
import numpy as np
warnings.simplefilter("ignore", np.ComplexWarning)

def set_verbose(is_verbose):
    """
    Choose whether or not to display output from calls to ``amitgroup.info``. 

    Parameters
    ----------
    is_verbose : bool
        If set to ``True``, info messages will be printed when running certain functions. Default is ``False``.
    """
    global _is_verbose
    _is_verbose = is_verbose

def info(*args, **kwargs):
    """
    Output info about the status of running functions. If you are writing a function that might take minutes to complete, periodic calls to the function with a description of the status is recommended.

    This function takes the same arguments as Python 3's ``print`` function. The only difference is that if ``amitgroup.set_verbose(True)`` has not be called, it will suppress any output. 
    """
    if _is_verbose:
        print(*args, **kwargs)

def warning(*args, **kwargs):
    """
    Output warning, such as numerical instabilities.
    """
    if not _is_silent:
        print("WARNING: " + args[0], *args[1:], **kwargs)


class AbortException(Exception):
    """
    This exception is used for when the user wants to quit algorithms mid-way. The `AbortException` can for instance
    be sent by pygame input, and caught by whatever is running the algorithm.
    """
    pass
