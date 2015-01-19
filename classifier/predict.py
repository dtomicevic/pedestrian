from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import product, imap, ifilter
from extractor import window
from utils.profiling import profile
import numpy as np


def model_predict(inputs, model):
    return model.predict(inputs)


class Detector(object):

    def __init__(self, model, filter, w):
        assert(w % 2 == 1)

        self.model = model
        self.filter = filter
        self.w = w

    @profile
    def predict(self, image):
        """ """
        d = (self.w - 1) / 2

        proc = (np.pad(self.filter(image), d, 'constant', constant_values=0) > 0).astype(int)

        org = np.copy(proc)

        print np.sum(org)
        print np.sum(proc)
        print np.sum((proc != org).astype(int))

        ii = np.where(proc != 0)

        samples = imap(lambda xy: window(proc, d, *xy), zip(*ii))

        proc[ii] = np.hstack(map(lambda s: self.model.predict(s), np.vstack(samples)))

        print np.sum(org)
        print np.sum(proc)
        print np.sum((proc != org).astype(int))

        return (org, proc, (proc != org).astype(int))


pool = Pool(cpu_count())
