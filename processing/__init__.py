from utils.profiling import profile
from functools import reduce, partial
import logging

logger = logging.getLogger(__name__)


def pipeline(filters):
    """ creates a pipeline for passing an image through a series of filters
        which accept an image as an argument and return the resulting image

        filters: list
            a list of filter objects in the order they will be applied in the
            filter chain

        returns: function
            a function representing a filter chain of ordered filters
    """
    return partial(reduce, lambda acc, f: f(acc), filters)


@profile
def process(dataset, f):
    """ adds an image generate using a filter chain to every sample in the
        dataset

        dataset: object
            object containing samples list compatible with the sample class
            interface

        f: function
            a filter chain generated using the pipeline function

        returns: None
            doesn't return anything. just updates the dataset sent as a
            function parameter
    """
    logger.info('processing dataset ({0})'.format(len(dataset.samples)))
    for sample in dataset.samples:
        sample.proc = f(sample.image)
