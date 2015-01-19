from os import listdir
from os.path import isfile, join
from parsers import pascal


class PennFudanDataset(object):
    """ represents a Penn-Fudan Pedestrian detection and segmentation dataset

        images from the dataset are taken from scenes around campus and urban
        street. objects of interest in these images are pedestrians. each image
        has at least one pedestrian in it.

        pedestrian height: [180, 390] px
        dataset size:      170 images
        pedestrian count:  345 pedestrians

        read more:
        http://www.cis.upenn.edu/~jshi/ped_html/
    """

    def __init__(self, path):
        """ loads dataset samples from PASCAL compatible annotation files
            given a directory path of the dataset

            path: string
                represents a path to the root directory of the downloaded
                dataset
        """
        if path is None:
            raise ValueError('Path can\'t be emtpy')

        # take position in the 'Annotation' folder
        path = join(path, 'Annotation')

        # retrieve filenames of all available PASCAL annotation files
        fnames = [f for f in listdir(path) if isfile(join(path, f))]

        # for each annotation file, generate a dataset sample from its
        # annotation filename using the PASCAL parser
        self.samples = map(lambda x: pascal.parse(join(path, x)), fnames)
