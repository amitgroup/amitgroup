
# Load the following modules by default
import io
import math
import ml
from deformation import ImageDeformation
from id_wavelet import IDWavelet

def __call__():
    print "test"

VERSION = (0,0,0)
ISRELEASED  = False
__version__ = '{0}.{1}.{2}'.format(*VERSION)
if not ISRELEASED:
    __version__ += '.dev' 

