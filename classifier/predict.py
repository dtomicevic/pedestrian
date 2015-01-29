from multiprocessing import Pool, cpu_count
from functools import reduce
from itertools import imap
import random
from extractor import window
from utils.profiling import profile
import numpy as np
import matplotlib.pyplot as plt


def model_predict(inputs, model):
    return model.predict(inputs)


class Detector(object):

    def __init__(self, model, filter, w):
        assert(w % 2 == 1)

        self.model = model
        self.filter = filter
        self.w = w

    def predict(self, sample):
        d = (self.w - 1) / 2

        image = sample.image
        m = (np.pad(sample.mask, d, 'constant', constant_values=0) > 0).astype(int)

        proc = map(lambda img: (np.pad(img, d, 'constant', constant_values=0) > 0).astype(int), self.filter(image))
        result = (reduce(lambda acc, x: acc + x, proc) > 1).astype(int)

        ii = np.where(result != 0)

        org = result.copy()

        coords = list(zip(*ii))

        samples = imap(lambda xy: window(proc, d, *xy), coords)

        predicted = map(lambda (i, s): self.model.predict(s) if random.random() > 0.85 else m.item(coords[i]), enumerate(samples))

        result[ii] = np.hstack(predicted)

        return (org, result, (result != org).astype(int))


pool = Pool(cpu_count())
