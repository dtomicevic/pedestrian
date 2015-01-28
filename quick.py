from dataset import PennFudanDataset
from processing import process
from classifier import extractor
from sklearn import svm
from filteropt import create_pipeline
from sklearn.externals import joblib
import argparse

parser = argparse.ArgumentParser(description='Train a single model')

parser.add_argument('filename', default=500, type=str, help='output filename')
parser.add_argument('-n', default=10000, type=int, help='number of samples')
parser.add_argument('-c', default=1.0, type=float, help='SVM C parameter')
parser.add_argument('-g', default=0.2, type=float, help='SVM gamma parameter')
parser.add_argument('-t', default=20, type=int, help='canny threshold')
parser.add_argument('-w', default=11, type=int, help='window size')
parser.add_argument('--cache', default=500, type=int, help='SVM cache')

args = parser.parse_args()

print('params: {0}'.format(args))

dataset = PennFudanDataset('dataset/PennFudanPed')
pipeline = create_pipeline(threshold=args.t)

process(dataset, pipeline)

inputs, targets = extractor.extract(dataset, w=args.w, N=args.n)

estimator = svm.SVC(C=args.c, gamma=args.g, cache_size=args.cache)

model = estimator.fit(inputs, targets)

joblib.dump(model, args.filename)
