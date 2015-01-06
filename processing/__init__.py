from functools import reduce


def pipeline(filters, cb=None):
    """ creates a pipeline for passing an image through a series of filters
        which accept an image as an argument and return the resulting image
    """
    if cb is not None and not callable(cb):
        raise ValueError('callback variable should be callable.')

    def reduce_filters(acc, f):
        filtered = f(acc)

        if cb is not None:
            cb(filtered)

        return filtered

    def process(image):
        return reduce(reduce_filters, filters, image)

    return process
