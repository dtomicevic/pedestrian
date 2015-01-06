import cv2


def create(threshold, ratio=3):
    """ creates a canny edge detection filter """

    def process(image):
        return cv2.Canny(image, threshold, threshold * ratio)

    return process
