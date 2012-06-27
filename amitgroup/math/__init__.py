
try:
    from interp2d import *
except ImportError:
    from interp2d_pure import *
    print("Warning: No Cython! Please compile amitgroup.")
