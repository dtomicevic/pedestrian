
class Sample(object):
    """ A dataset sample containing a color image, a multivalue mask marking
        pixels representing pedestrians in the image and a list of bouding
        boxes surrounding pedestrians in the image
    """

    def __init__(self, image=None, mask=None, bboxes=None):
        self.image = image,
        self.mask = mask,
        self.bboxes = bboxes if bboxes is not None else []
