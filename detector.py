from dataset import PennFudanDataset
from sklearn.externals import joblib
from classifier.predict import Detector
from filteropt import create_pipeline
import matplotlib.pyplot as plt
import cv2

dataset = PennFudanDataset('dataset/PennFudanPed')

# model = joblib.load('models/quick/quick.pkl')

# detector = Detector(model, create_pipeline(threshold=20), w=11)


def threshold_cb(thresh):
    f = create_pipeline(threshold=thresh)
    r, g, b = f(dataset.samples[0].image)
    cv2.imshow('B', b)
    cv2.imshow('G', g)
    cv2.imshow('R', r)

cv2.namedWindow('B', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('G', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('R', cv2.WINDOW_AUTOSIZE)

t = 20

cv2.createTrackbar('canny thresh:', 'B', t, 255, threshold_cb)

threshold_cb(t)
cv2.waitKey(0)
cv2.destroyAllWindows()

# for sample in dataset.samples:
    

#     cv2.imshow('R', r)
#     cv2.imshow('G', g)
#     cv2.imshow('B', b)
#     plt.imshow(r, cmap='Greys',  interpolation='nearest')
#     plt.show()
#     exit(0)

# for sample in dataset.samples:
#     org, predicted, difference = detector.predict(sample.image)

#     plt.subplot(221)
#     plt.imshow(sample.mask, cmap='Greys',  interpolation='nearest')
#     plt.subplot(222)
#     plt.imshow(org, cmap='Greys',  interpolation='nearest')
#     plt.subplot(223)
#     plt.imshow(predicted, cmap='Greys',  interpolation='nearest')
#     plt.subplot(224)
#     plt.imshow(difference, cmap='Greys',  interpolation='nearest')
#     plt.show()
#     exit(0)
