
class ImageDeformation(object):
    """
    Image deformation. Base class for representing a deformation of a 2D mesh grid. 
    """
    def deform_map(self, x, y):
        raise NotImplemented("Can't use ImageDeformation directly")

    def deform(self, I):
        raise NotImplemented("Can't use ImageDeformation directly")
