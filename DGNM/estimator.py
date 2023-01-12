"""
@author: Jesús Prada Alonso
"""

import numpy as np
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from skmetrics import geometric_roc_auc_score
from skopt import BayesSearchCV

from dsvm import DSVC, DSVR, Straight


def select_estimator(name, cv=None, sigma=0.1, d=1):
    """ Select an estimator. """
    scoring = {'geometric_roc_auc': make_scorer(geometric_roc_auc_score, needs_proba=True),
               'mean_absolute_error': make_scorer(mean_absolute_error)}
    epsilons = sigma * np.logspace(-6, 3, base=2.0)
    gammas = np.logspace(-3, 6, base=2.0) / d
    dsvc = lambda n: BayesSearchCV(Pipeline([('scaler', StandardScaler()), ('classifier', DSVC(architecture=Straight(dense_units=[100]*n), epochs=100))]),
                                   {'classifier__kernel_regularizer_l2': [10**-6, 10**3, 'log-uniform'],
                                    'classifier__dense_kernel_regularizer_l2': [10**-6, 10**3, 'log-uniform']},
                                   scoring=scoring['geometric_roc_auc'], cv=cv, error_score=np.nan, fit_params={'classifier__epochs': 100})
    dsvr = lambda n: BayesSearchCV(Pipeline([('scaler', StandardScaler()), ('regressor', DSVR(architecture=Straight(dense_units=[100]*n), epochs=100))]),
                                   {'regressor__kernel_regularizer_l2': [10**-6, 10**3, 'log-uniform'],
                                    'regressor__dense_kernel_regularizer_l2': [10**-6, 10**3, 'log-uniform'],
                                    'regressor__loss_epsilon': epsilons},
                                   scoring=scoring['mean_absolute_error'], cv=cv, error_score=np.nan, fit_params={'regressor__epochs': 100})
    estimator = {'SVC': BayesSearchCV(Pipeline([('scaler', StandardScaler()), ('classifier', SVC(probability=True))]),
                                      {'classifier__C': [10**-3, 10**6, 'log-uniform'],
                                       'classifier__gamma': gammas},
                                      scoring=scoring['geometric_roc_auc'], cv=cv, error_score=np.nan),
                 'SVR': BayesSearchCV(Pipeline([('scaler', StandardScaler()), ('regressor', SVR())]),
                                      {'regressor__C': [10**-3, 10**6, 'log-uniform'],
                                       'regressor__epsilon': epsilons,
                                       'regressor__gamma': gammas},
                                      scoring=scoring['mean_absolute_error'], cv=cv, error_score=np.nan),
                 'DSVC0': dsvc(0), 'DSVC1': dsvc(1), 'DSVC2': dsvc(2), 'DSVC3': dsvc(3), 'DSVC4': dsvc(4), 'DSVC5': dsvc(5),
                 'DSVR0': dsvr(0), 'DSVR1': dsvr(1), 'DSVR2': dsvr(2), 'DSVR3': dsvr(3), 'DSVR4': dsvr(4), 'DSVR5': dsvr(5)}
    return estimator[name]
