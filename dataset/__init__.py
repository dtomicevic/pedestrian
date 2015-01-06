from os import listdir
from os.path import isfile, join
from parsers import pascal


class PennFudanDataset(object):

    def __init__(self, path=None):
        self.samples = []

        if path is None:
            raise ValueError('Path can\'t be emtpy')

        path = join(path, 'Annotation')

        fnames = [f for f in listdir(path) if isfile(join(path, f))]

        for fname in fnames:
            self.samples.append(pascal.parse(join(path, fname)))
