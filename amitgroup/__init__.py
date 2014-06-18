from __future__ import absolute_import

# Load the following modules by default
from .core import set_verbose, info, warning, AbortException

# Lazy load these?
import amitgroup.io
import amitgroup.features
import amitgroup.stats
import amitgroup.util
import amitgroup.plot
from . import image

VERSION = (0, 0, 0)
ISRELEASED = False
__version__ = '{0}.{1}.{2}'.format(*VERSION)
if not ISRELEASED:
    __version__ += '.dev'


def test(verbose=False):
    import amitgroup.tests
    import unittest
    unittest.main()
