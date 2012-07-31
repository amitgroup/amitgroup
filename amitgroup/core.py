
from __future__ import print_function

import amitgroup as ag
import sys

# If any of the arguments is '--verbose', set this to true
_is_verbose = '--verbose' in sys.argv
_is_silent = '--silent' in sys.argv

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
