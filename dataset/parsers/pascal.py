import cv2
import re
import logging
from os.path import join
from dataset.sample import Sample
from dataset.boundingbox import BoundingBox

logger = logging.getLogger(__name__)


def parse_image(line):
    """ parses the string formated as the PASCAL image filename annotation

        e.g. Image filename : "PennFudanPed/PNGImages/FudanPed00001.png"

        line: string
            line in the PASCAL annotation file with image information

        returns: array
            an image array parsed from the given line
    """
    filename = line.split(':')[-1].replace('"', '').strip()
    return cv2.imread(join('dataset', filename))


def parse_mask(line):
    """ parses the string formated as the PASCAL pixel mask annotation

        e.g. Pixel mask for object 1 "PASpersonWalking" :
            "PennFudanPed/PedMasks/FudanPed00001_mask.png"

        line: string
            line in the PASCAL annotation file with mask information

        returns: array
            a mask array parsed from the given line
    """
    filename = line.split(':')[-1].replace('"', '').strip()
    mask = cv2.imread(join('dataset', filename), cv2.CV_LOAD_IMAGE_GRAYSCALE)
    return cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]


def parse_bbox(line):
    """ parses the string formated as the PASCAL bounding box annotation

        e.g. Bounding box for object 2 "PASpersonWalking"
                (Xmin, Ymin) - (Xmax, Ymax) : (420, 171) - (535, 486)

        line: string
            line in the PASCAL annotation file with bounding box information

        returns: BoundingBox
            a bounding box parsed from the given line
    """
    values = [int(s) - 1 for s in re.findall(r"[\w']+", line.split(':')[-1])]
    return BoundingBox.fromlist(values)


def parse(filename):
    """ parses the PASCAL compatible annotation file

    filename: string
        path to the annotation file

    returns: Sample
        instance of the sample class parsed from th e annotation file
    """
    if not filename:
        raise ValueError('filename can\'t be emtpy')

    sample = Sample()
    logger.info(filename)

    with open(filename) as f:
        for line in f:
            if sample.image is not None and line.startswith('Image filename'):
                sample.image = parse_image(line)
                continue

            if sample.mask is not None and line.startswith('Pixel mask'):
                sample.mask = parse_mask(line)
                continue

            if line.startswith('Bounding box'):
                sample.bboxes.append(parse_bbox(line))

    return sample
