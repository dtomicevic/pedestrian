import cv2


def create(d=11, sigma_color=75, sigma_space=75):
    """ creates a bilateral filter function """

    def process(image):
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    return process
