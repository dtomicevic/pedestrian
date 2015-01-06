import cv2
import re
import logging
from os.path import join
from dataset.sample import Sample
from dataset.boundingbox import BoundingBox

logger = logging.getLogger(__name__)


def parse_image(line):
    filename = line.split(':')[-1].replace('"', '').strip()
    return cv2.imread(join('dataset', filename))


def parse_mask(line):
    filename = line.split(':')[-1].replace('"', '').strip()
    mask = cv2.imread(join('dataset', filename), cv2.CV_LOAD_IMAGE_GRAYSCALE)
    return cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]


def parse_bbox(line):
    values = [int(s) - 1 for s in re.findall(r"[\w']+", line.split(':')[-1])]
    return BoundingBox.fromlist(values)


def parse(filename):
    """ parses the PASCAL compatible annotation file """
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
