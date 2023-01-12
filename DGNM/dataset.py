"""
@author: Jesús Prada Alonso
"""

from skdatasets.libsvm import load as load_libsvm
from sklearn.model_selection import PredefinedSplit


def select_dataset(name):
    """ Select a dataset. """
    if name in ['abalone', 'bodyfat', 'cpusmall', 'housing', 'mg', 'mpg',
                'pyrim', 'space_ga', 'triazines']:
        X, y = load_libsvm[name](return_X_y=True)
        X_test = y_test = None
        inner_cv = 3
        outer_cv = 3
    elif name in ['a4a', 'a8a', 'cod-rna', 'combined', 'epsilon', 'news20',
                  'pendigits', 'usps', 'w7a', 'w8a']:
        (X, y), (X_test, y_test) = load_libsvm[name](return_X_y=True)
        y[y == -1] = 0
        y_test[y_test == -1] = 0
        inner_cv = 3
        outer_cv = None
    elif name in ['dna', 'ijcnn1', 'letter', 'satimage', 'shuttle']:
        (X, y), (X_tr, y_tr), (X_val, y_val), (X_test, y_test) = load_libsvm[name](return_X_y=True)
        y[y == -1] = 0
        y_test[y_test == -1] = 0
        inner_cv = PredefinedSplit([item for sublist in [[-1] * len(X_tr), [0] * len(X_val)] for item in sublist])
        outer_cv = None
    else:
        raise Exception('Dataset not available')
    return X, y, X_test, y_test, inner_cv, outer_cv
