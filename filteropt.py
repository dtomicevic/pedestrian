from dataset import PennFudanDataset
from processing import process, pipeline
from processing.filters import grayscale, bilateral, canny
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import train_test_split
from classifier import extractor
from sklearn import svm
from utils.profiling import profile
from itertools import product
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset = PennFudanDataset('dataset/PennFudanPed')


def create_pipeline(threshold=15):
    """creates a filter pipeline using supplied arguments

        threshold: int
            canny filter threshold value (see canny in opencv)

        return: function
            function containing and entire pipeline of chained filters
    """
    return pipeline([grayscale(), bilateral(), canny(threshold)])


@profile
def train_model(inputs, targets):
    """ splits the given data samples into a training and test set and trains
        an SVM model using that data. trained model is then evaluated on the
        test set and precision, recall and fscore are calculated for that
        model on the test set

        inputs: array-like
            input array for the SVM

        targets: array-like
            target array for the SVM, must have the same shape[0]

        returns: tuple
            a tuple containing precision, recall and f1 score for the trained
            model
    """
    assert(inputs.shape[0] == targets.shape[0])

    # splits the dataset into training and validations ets with ratio 1/7 ratio
    # for training set and 6/7 ratio for validation set
    x_train, x_test, y_train, y_test = \
        train_test_split(inputs, targets, test_size=6/7.0)

    # trains an SVM model with default parameters and rbf kernel. in this stage
    # we're not worried about totally optimiozing the SVM model. default
    # settings work fine. once we optimize filter parameters, SVM grid search
    # will come to place to select the best model for chosen filter parameters
    model = svm.SVC(kernel='rbf')
    model.fit(x_train, y_train)

    # test the trained model on the test data and report back precision, recall
    # and f1 scores
    predicted = model.predict(x_test)

    score = precision_recall_fscore_support(y_test, predicted, average='macro')
    return score[:3]


if __name__ == '__main__':
    # this loop implements grid search over viable parameters for canny
    # threshold and window size to find the best one
    for i, (c, w) in enumerate(product(xrange(5, 100, 5), [7, 11, 15])):
        logger.info('iteration {0}: canny={1}, window={2}'.format(i, c, w))

        process(dataset, create_pipeline(threshold=c))
        inputs, targets = extractor.extract(dataset, w=w, N=70000)
        score = train_model(inputs, targets)

        print('precision={0:.4f} recall={1:.4f}, f1={2:.4f}'.format(*score))
