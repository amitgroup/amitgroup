
try:
    from interp2d import *
except ImportError:
    from interp2d_pure import *
    print("Warning: No Cython! Please compile amitgroup.")

from blur import blur_image

__all__ = ['interp2d', 'blur_image']
