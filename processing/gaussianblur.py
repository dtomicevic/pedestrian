import cv2


def create(kernel_size, stddev=0):
    """ creates a gaussian blur filter """

    def process(image):
        return cv2.GaussianBlur(image, kernel_size, stddev)

    return process
