from dataset import PennFudanDataset
from sklearn.externals import joblib
from classifier.predict import Detector
from filteropt import create_pipeline
import matplotlib.pyplot as plt

dataset = PennFudanDataset('dataset/PennFudanPed')

model = joblib.load('models/quick/quick.pkl')

detector = Detector(model, create_pipeline(threshold=20), w=11)

for sample in dataset.samples:
    org, predicted, difference = detector.predict(sample.image)

    plt.subplot(221)
    plt.imshow(sample.mask, cmap='Greys',  interpolation='nearest')
    plt.subplot(222)
    plt.imshow(org, cmap='Greys',  interpolation='nearest')
    plt.subplot(223)
    plt.imshow(predicted, cmap='Greys',  interpolation='nearest')
    plt.subplot(224)
    plt.imshow(difference, cmap='Greys',  interpolation='nearest')
    plt.show()
    exit(0)
