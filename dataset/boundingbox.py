

class BoundingBox(object):
    """ A rectangle surrounding an object in the image defined by two points on
        opposite corners of the rectangle
    """

    def __init__(self, a, b):
        """ creates a bounding box from two points on opposite corners of the
            bounding box

            a: tuple of ints
                coordinates of the first point

            a: tuple of ints
                coordinates of the second point
        """
        self.a = a
        self.b = b

    @classmethod
    def fromlist(cls, values):
        """ creates a bounding box from a list of points

            values: list of tuples
                list containing tuples representing points of the opposite
                corners on the bounding box
        """
        return cls(*zip(*2*[iter(values)]))

    @property
    def area(self):
        """ computes the area of the bounding box

            area = |x2 - x1| * |y2 - y1|

            returns: int
                area of the bounding box
        """
        return abs(self.a[0] - self.b[0]) * abs(self.a[1] - self.b[1])

    @property
    def center(self):
        """ computes the center of the bounding box

                      x1 + x2  y1 + y2
            center = (-------, -------)
                         2        2

            returns: tuple
                contains x and y coordinates of the bounding box center
        """
        return ((self.a[0] + self.b[0]) / 2, (self.a[1] + self.b[1]) / 2)
