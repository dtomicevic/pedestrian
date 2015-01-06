

class BoundingBox(object):
    """ A rectangle surrounding an object in the image defined by two points on
        opposite corners of the rectangle
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    @classmethod
    def fromlist(cls, values):
        return cls(*zip(*2*[iter(values)]))

    @property
    def area(self):
        return abs(self.a[0] - self.b[0]) * abs(self.a[1] - self.b[1])

    @property
    def center(self):
        return ((self.a[0] + self.b[0]) / 2, (self.a[1] + self.b[1]) / 2)
