from dataset import PennFudanDataset
from sklearn.externals import joblib
from classifier.predict import Detector
from filteropt import create_pipeline
import numpy as np
import cv2

dataset = PennFudanDataset('dataset/PennFudanPed')

for sample in dataset.samples:
    print('processing {0}'.format(sample.name))
    model = joblib.load('trained/quick_2.pkl')
    detector = Detector(model, create_pipeline(threshold=20), w=11)
    org, proc, diff = detector.predict(sample)
    proc = np.array(proc * 255, dtype=np.uint8)
    cv2.imwrite('detected/{0}.png'.format(sample.name), proc)
