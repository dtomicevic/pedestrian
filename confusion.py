from sklearn.externals import joblib
from dataset import PennFudanDataset
from processing import process
from classifier import extractor
from filteropt import create_pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

dataset = PennFudanDataset('dataset/PennFudanPed')
pipeline = create_pipeline(threshold=20)

process(dataset, pipeline)

inputs, targets = extractor.extract(dataset, w=11, N=20000)

model = joblib.load('trained/quick_2.pkl')

predicted = model.predict(inputs)

cm = confusion_matrix(targets, predicted)

print(cm)

plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
