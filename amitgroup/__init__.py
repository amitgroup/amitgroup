from __future__ import absolute_import

# Load the following modules by default
from .core import *
import amitgroup.io

VERSION = (0,0,0)
ISRELEASED  = False
__version__ = '{0}.{1}.{2}'.format(*VERSION)
if not ISRELEASED:
    __version__ += '.dev' 

