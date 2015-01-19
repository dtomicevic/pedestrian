
class Sample(object):
    """ A dataset sample containing a color image, a multivalue mask marking
        pixels representing pedestrians in the image and a list of bounding
        boxes surrounding pedestrians in the image
    """

    def __init__(self, image=None, mask=None, bboxes=None):
        """ create a sample from a pedestrian image, image mask and a list
            of bounding boxes

            image: color array
                a two dimensional color array representing the image the sample
                represents

            mask: binary array
                a two dimensional binary array with values greater than zero
                representing pixels containing a pedestrian and zero values
                otherwise

            bboxes: list of BoundingBox
                a list containing bounding boxes for pedestrian on the image
        """
        self.image = image,
        self.mask = mask,
        self.bboxes = bboxes if bboxes is not None else []
