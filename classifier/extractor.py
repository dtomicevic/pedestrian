from itertools import product, repeat, chain, ifilter, imap
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import binarize
from utils.profiling import profile
from numpy.random import randint
from functools import partial
from random import sample
import numpy as np
import logging

logger = logging.getLogger(__name__)


def on_edge(mask, x, y):
    """ checks if the point defined with coordinates x and y lies on the edge
        of a pedestrian on the pedestrian mask

        to lie on an edge is defined by having at least one non-pedestrian
        pixel in the surrouding of the given pixel (left, right, up or down)

        e.g. we're checking if the middle pixel is an edge one

            0 1 1
        ... 0 1 1 ... -> on_edge? True
            0 0 1

            0 1 1
        ... 1 1 1 ... -> on_edge? False
            0 1 1

        mask: array-like
            two dimensional array of binarized pixels

        x: int
            point coordinate on the x-axis

        y: int
            point coordinate on the y-axis

        returns: bool
            a boolean stating weather the point lies on the edge of the
            pedestrian
    """
    return mask[x, y] and not all([mask[x + dx, y + dy] for dx, dy in
                                   zip([-1, 1, 0, 0], [0, 0, -1, 1])])


def window(image, d, x, y):
    """ extracts a window of size d pixels in each direction from the point
        defined by x and y coordinates

        total number of pixels in the window is (2 * d + 1)^2

        window is binarized if not already binary and reshaped to a vector

        e.g. d = 2, P(x, y)

        0   0 0 0 0 0   0 0 0 0
           -----------
        0 | 0 0 0 0 0 | 0 0 1 1
        0 | 0 0 0 0 0 | 1 1 1 1
        0 | 0 0 P 0 1 | 1 1 1 1
        0 | 0 0 0 0 1 | 1 1 1 1
        0 | 0 0 0 1 1 | 1 1 1 1
           -----------
        0   0 0 1 1 1   1 1 1 1
        0   0 0 0 1 1   1 1 1 1
        0   0 0 1 1 1   1 1 1 1

        image: array-like
            two dimensional array

        d: int
            number of pixels to take in each direction from the center

        x: int
            x coordinate of the window center

        y: int
            x coordinate of the window center

        returns: array
            a binary array with (2 * d + 1)^2 elements
    """
    w = image[(x - d):(x + d + 1), (y - d):(y + d + 1)]
    return binarize(w, 0.5).reshape(-1)


def samples(s, d):
    """ generates classifier input samples from a dataset sample

        s: object
            compatible with the sample class interface

        d: int
            number of pixels to take in each direction from the center when
            generating an input sample via the moving window

        return: list of tuples
            tuples contain two elements, an input vector and a target vector
            extracted from a given sample
    """

    # generate a cartesian product of all possible point coordinates given
    # image shape and offset d
    # filter out all points not representing pedestrian mask edges
    # compute input vectors for all remaining points from their respective
    # windows
    positive = imap(lambda xy: (window(s.proc, d, *xy), 1),
                    ifilter(lambda xy: on_edge(s.mask, *xy),
                    product(*map(lambda x: xrange(d, x - d),
                            s.proc.shape))))

    # create an infinite uniform random sampling list of point coordinates
    # inside the given image
    # filter out all points representing positive examples to get an infinite
    # list of point coordinates representing negative examples
    # compute input vectors for all points from their respective windows
    negative = imap(lambda xy: (window(s.proc, d, *xy), 0),
                    ifilter(lambda xy: not s.mask.item(*xy),
                    imap(lambda o: map(lambda x: randint(d, x - d), o),
                         repeat(s.proc.shape))))

    # zip a finite list of positive examples and an infinite list of negative
    # examples to get an equal amount of positive and negative examples and has
    # a length of len(positive)
    # chain all the zipped elements to get a flattened list of examples
    # containing both positive and negative examples in one list
    return list(chain(*zip(positive, negative)))


def generate(dataset, w):
    """ generate a list of classifier data samples from all dataset samples
        with a parallel implementation using a thread pool

        dataset: object
            object containing samples list compatible with the sample class
            interface

        w: int
            size of the window used to extract features from an image
            must be an odd number

        returns: iterator
            iterator contains all the positive and negative data samples
            generated from the dataset
    """
    logger.info('extracting samples using {0} threads'.format(cpu_count()))
    return chain(*pool.map(partial(samples, d=(w - 1) / 2), dataset.samples))


@profile
def extract(dataset, w=11, N=25000):
    """ extracts the training inputs and targets from the dataset

        dataset: object
            object containing samples list compatible with the sample class
            interface

        w: int
            size of the window used to extract features from an image
            must be an odd number

        N: int
            the number of samples to extract from the dataset. samples are
            extracted randomly from the list of all possible samples
            must be positive

        returns: tuple of numpy arrays
            the tuple contains two numpy arrays, one represents an input two
            dimensional array and the other one represents a target vector
    """
    assert(w % 2 == 1)
    assert(N > 0)

    # generates a list of data samples used in the model training
    #
    # randomly samples the list of samples and returns a maximum of
    # N data samples as tuples of (input, target) vectors
    #
    # zips the sample tuples to divide input vectors in a separate tuple and
    # target vectors in a separate tuple
    inputs, targets = zip(*sample(list(generate(dataset, w)), N))

    # vertically concatenates list of numpy arrays and concatenates a list
    # of target vectors to a numpy array
    return (np.vstack(inputs), np.array(targets))


# process pool for concurrent sample generation
pool = Pool(cpu_count())
