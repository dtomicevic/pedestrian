import cv2


def create(conversion_code=cv2.COLOR_BGR2GRAY):
    """ creates a color to grayscale conversion filter """

    def process(image):
        return cv2.cvtColor(image, conversion_code)

    return process
