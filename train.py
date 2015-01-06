import cv2
import numpy as np
import logging
from dataset import PennFudanDataset
from dataset.parsers import pascal
import matplotlib.pyplot as plt
from processing import pipeline, grayscale, gaussianblur, canny, contours, clahehist, bilateral
from classifier import extractor
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

logging.basicConfig(level=logging.INFO)

# dataset = PennFudanDataset('dataset/PennFudanPed')


sample = pascal.parse('dataset/PennFudanPed/Annotation/FudanPed00001.txt')

fgr = grayscale.create()
fbi = bilateral.create()
fch = clahehist.create()
fgb = gaussianblur.create((7, 7))
fcn = canny.create(85)
fct = contours.create()

f = pipeline([fgr, fbi, fcn, fct])

cv2.imshow('input', sample.image)
cv2.imshow('output', f(sample.image))
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)

# for i, sample in enumerate(dataset.samples):
#     print('Processing {0}/{1}'.format(i, len(dataset.samples)))
#     sample.processed = sample.mask  # f(sample.image)

# inputs, targets = extractor.extract(dataset)

inputs = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
targets = np.array([0, 1, 1, 1])

model = svm.NuSVC(nu=0.1, kernel='rbf')
# model = LogisticRegression()
model.fit(inputs, targets)

predicted = model.predict(inputs)

cm = confusion_matrix(targets, predicted)

print(cm)

score = precision_recall_fscore_support(targets, predicted, average='macro')
print(score)

plt.matshow(cm)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# def draw():
#     stages = []

#     def drawcb(filtered):
#         cv2.imshow('{0}'.format(len(stages)), filtered)
#         stages.append(1)

#     f = pipeline([fgr, fbi, fcn, fct], drawcb)
#     f(image)


# def canny_cb(thresh):
#     global fcn
#     fcn = canny.create(thresh)
#     draw()


# cv2.namedWindow('input', cv2.WINDOW_AUTOSIZE)
# cv2.createTrackbar('canny thresh:', 'input', 55, 255, canny_cb)
# cv2.imshow('input', image)

# draw()
# cv2.destroyAllWindows()
