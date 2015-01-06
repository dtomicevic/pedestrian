import numpy as np
import cv2


def create(kernel_size, **kwargs):
    """ creates an erosion filter """
    kernel = np.ones(kernel_size, np.uint8)

    def process(image):
        return cv2.erode(image, kernel, **kwargs)

    return process
