import numpy as np
import cv2


def create(**kwargs):
    """ creates a filter for finding contours in a grayscale image """
    mode = kwargs.get('mode', cv2.RETR_EXTERNAL)
    method = kwargs.get('method', cv2.CHAIN_APPROX_SIMPLE)
    contour_idx = kwargs.get('contour_idx', -1)
    color = kwargs.get('color', (255, 255, 255))

    def process(image):
        contours, hierarchy = cv2.findContours(image, mode, method)
        contoured_image = np.zeros(image.shape, np.uint8)

        for contour in contours:
            cv2.drawContours(contoured_image, contours, contour_idx, color, thickness=1)

        return contoured_image

    return process
