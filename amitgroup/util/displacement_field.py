
class DisplacementField(object):
    """
    Displacement field. Base class for representing a deformation of a 2D mesh grid. 
    """
    def __init__(self, shape):
        self.set_shape(shape)

    def set_shape(self, shape):
        pass 

    def deform_map(self, x, y):
        raise NotImplemented("Can't use ImageDeformation directly")

    def deform(self, I):
        raise NotImplemented("Can't use ImageDeformation directly")
