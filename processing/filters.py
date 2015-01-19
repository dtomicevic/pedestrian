from functools import partial
import cv2


def bilateral(d=11, c=75, s=75):
    """ creates a bilateral blur filter function

        bilateralFilter can reduce unwanted noise very well while keeping edges
        fairly sharp. it's, however, much slower than other methods

        d: int
            diameter of each pixel neighborhood that is used during filtering
            if it is non-positive, it is computed from sigmaSpace

        c: int
            filter sigma in the color space
            a larger value of the parameter means that farther colors within
            the pixel neighborhood will be mixed together, resulting in larger
            areas of semi-equal color

        returns: function
            a bilateral filter function with specialized arguments
    """
    return partial(cv2.bilateralFilter, d=d, sigmaColor=c, sigmaSpace=s)


def canny(hysthres=15, ratio=3):
    """ creates a canny edge detection filter function

        hysthres: int
            first threshold for the hysteresis procedure
            value in the interval [0, 255]

        ratio: int
            used to calculate second threshold for the hysteresis procedure
            threshold2 = hysthres * ratio
            rule of thumb suggests to keep use 3 as the ration value

        returns: function
            a canny edge detection function with specialized arguments
    """
    return partial(cv2.Canny, threshold1=hysthres, threshold2=hysthres * ratio)


def grayscale(code=cv2.COLOR_BGR2GRAY):
    """ creates a color to grayscale conversion filter

        code: enum
            a conversion code flag

        returns: function
            a grayscale conversion function
    """
    return partial(cv2.cvtColor, code=code)
