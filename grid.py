import logging
from dataset import PennFudanDataset
from processing import process
from classifier import extractor
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from multiprocessing import cpu_count
from filteropt import create_pipeline
from sklearn.externals import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def grid_search(grid, inputs, targets, threads=cpu_count(), verbose=3):
    estimator = svm.SVC(cache_size=2000)
    model = GridSearchCV(estimator, grid, n_jobs=threads, verbose=verbose)
    model.fit(inputs, targets)
    return model.best_estimator_


def generate_sets(dataset, w=11, N=50000):
    process(dataset, create_pipeline())
    return extractor.extract(dataset, w=w, N=N)


if __name__ == '__main__':
    grid = {
        'kernel': ['rbf'],
        'C': [2e-5, 2e-3, 2e-1, 1, 2e1, 2e3, 2e5],
        'gamma': [2e-11, 2e-7, 2e-3, 2e-1, 1, 2e1, 2e3]
    }

    dataset = PennFudanDataset('dataset/PennFudanPed')

    # generate train and test sets
    inputs, targets = generate_sets(dataset)

    # search for the best estimator using grid search
    best = grid_search(grid, inputs, targets)

    # print the params of the best estimator found using grid search
    logger.log(best)

    # dump the best estimator to file for further use
    joblib.dump(best, 'grid_best.pkl')
