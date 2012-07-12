from __future__ import absolute_import

# Load the following modules by default
from .core import *
import amitgroup.io
import amitgroup.math
import amitgroup.ml
from .deformation import ImageDeformation
from .id_wavelet import IDWavelet

VERSION = (0,0,0)
ISRELEASED  = False
__version__ = '{0}.{1}.{2}'.format(*VERSION)
if not ISRELEASED:
    __version__ += '.dev' 

