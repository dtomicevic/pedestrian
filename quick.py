from dataset import PennFudanDataset
from processing import process
from classifier import extractor
from sklearn import svm
from filteropt import create_pipeline
from sklearn.externals import joblib

dataset = PennFudanDataset('dataset/PennFudanPed')
pipeline = create_pipeline(threshold=20)

process(dataset, pipeline)

inputs, targets = extractor.extract(dataset, w=11, N=15000)

estimator = svm.SVC(C=1.0, gamma=0.2, cache_size=700)

model = estimator.fit(inputs, targets)

joblib.dump(model, 'models/quick/quick.pkl')
