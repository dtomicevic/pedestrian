import cv2


def create(**kwargs):
    """ creates a contrast limited adaptive histogram equalization filter """
    clahe = cv2.createCLAHE(**kwargs)

    def process(image):
        return clahe.apply(image)

    return process
