"""
@author: Jesús Prada Alonso
"""

import numpy as np
from sacred import Experiment, Ingredient
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
import tempfile

from dataset import select_dataset
from estimator import select_estimator


dataset = Ingredient('dataset')
select_dataset = dataset.capture(select_dataset)
estimator = Ingredient('estimator')
select_estimator = estimator.capture(select_estimator)
experiment = Experiment(ingredients=[dataset, estimator])
experiment.add_config(persist=False)


@experiment.automain
def run(persist):
    """ Run an experiment. """
    X, y, X_test, y_test, inner_cv, outer_cv = select_dataset()
    estimator = select_estimator(cv=inner_cv, sigma=np.std(y), d=y.shape[1] if len(y.shape) >= 2 else 1)
    if (X_test is not None) and (y_test is not None):
        # Test score
        estimator.fit(X, y)
        scores = estimator.score(X_test, y_test)
    else:
        # CV score
        scores = cross_val_score(estimator, X, y, cv=outer_cv)
        if persist:
            estimator.fit(X, y)
    experiment.info['scores'] = scores
    if persist:
        handler = tempfile.NamedTemporaryFile('wb')
        joblib.dump(estimator, handler)
        experiment.add_artifact(handler.name, name='estimator.pkl')
    return scores
