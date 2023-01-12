"""
@author: Jesús Prada Alonso
"""

from functools import partialmethod
from keras import backend as K
from skkeras import FFClassifier, FFRegressor, Straight


class DSVC(FFClassifier):

    __init__ = partialmethod(FFClassifier.__init__, loss='hinge')

    fit = partialmethod(FFClassifier.fit, loss='hinge')


class DSVR(FFRegressor):

    def __init__(self, architecture=None, activation='linear', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer_l1=None, kernel_regularizer_l2=None,
                 bias_regularizer_l1=None, bias_regularizer_l2=None,
                 activity_regularizer_l1=None, activity_regularizer_l2=None,
                 kernel_constraint=None, bias_constraint=None, optimizer='adam',
                 lr=0.001, momentum=0.0, nesterov=False, decay=0.0, rho=0.9,
                 epsilon=1e-08, beta_1=0.9, beta_2=0.999, schedule_decay=0.004,
                 loss='mse', metrics=None, loss_weights=None,
                 sample_weight_mode=None, batch_size='auto', epochs=200,
                 verbose=2, early_stopping=True, tol=0.0001, patience=2,
                 validation_split=0.1, validation_data=None, shuffle=True,
                 class_weight=None, sample_weight=None, initial_epoch=0,
                 window=None, return_sequences=False, loss_epsilon=0.1):
        for k, v in locals().items():
            if k != 'self':
                self.__dict__[k] = v

    def _epsilon_insensitive(self, y_true, y_pred):
        return K.mean(K.maximum(K.abs(y_pred - y_true) - self.loss_epsilon,
                                0.0), axis=-1)

    def fit(self, X, y, **kwargs):
        return FFRegressor.fit(self, X, y, loss=self._epsilon_insensitive,
                               **kwargs)
