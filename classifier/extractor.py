from functools import reduce
from itertools import product
from operator import mul
import logging
import numpy as np

logger = logging.getLogger(__name__)


def extract(dataset, w=11, step=220):
    assert(w % 2 == 1)
    assert(step > 0)

    def count(shape):
        return reduce(mul, map(lambda x: (x - w) // step + 1, shape), 1)

    n = sum(map(lambda x: count(x.processed.shape), dataset.samples))
    logger.info('Total samples to extract: {0}'.format(n))

    def generate(dataset):
        for i, sample in enumerate(dataset.samples):
            logger.info('Extracting samples from {0}'.format(i))
            s = sample.processed.shape

            for x, y in product(*map(lambda x: range(0, x - w + 1, step), s)):
                data = sample.processed[x:(x + w), y:(y + w)]
                mask = sample.mask[(x + w - 1) / 2, (y + w - 1) / 2]
                yield (data, mask)

    inputs = np.zeros((n, w * w))
    targets = np.zeros(n)

    for i, (data, mask) in enumerate(generate(dataset)):
        inputs[i] = data.reshape(-1)
        targets[i] = 0 if mask == 0 else 1

    print np.sum(targets)

    return (inputs, targets)
