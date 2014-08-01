from __future__ import absolute_import

# Load the following modules by default
from .core import (set_verbose,
                   info,
                   warning,
                   AbortException,
                   bytesize,
                   humanize_bytesize,
                   memsize,
                   apply_once_over_axes,
                   Timer)


# Lazy load these?
import amitgroup.io
import amitgroup.features
import amitgroup.stats
import amitgroup.util
import amitgroup.plot
from . import image

VERSION = (0, 9, 0)
ISRELEASE = True
__version__ = '{0}.{1}.{2}'.format(*VERSION)
if not ISRELEASE:
    __version__ += '.dev'


def test(verbose=False):
    import amitgroup.tests
    import unittest
    unittest.main()
