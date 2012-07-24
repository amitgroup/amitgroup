from __future__ import absolute_import

# Load the following modules by default
from .core import *

# Lazy load these?
import amitgroup.io
import amitgroup.features
import amitgroup.ml
import amitgroup.stats
import amitgroup.plot

VERSION = (0,0,0)
ISRELEASED  = False
__version__ = '{0}.{1}.{2}'.format(*VERSION)
if not ISRELEASED:
    __version__ += '.dev' 

