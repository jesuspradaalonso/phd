#!/home/proyectos/ada2/local/anaconda2/bin/python
import numpy as np
import sys

#import theano.sandbox.cuda
#theano.sandbox.cuda.use('gpu1')
from scipy.io import loadmat
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.regularizers import l2, activity_l2

from pylab import *
C       = 100000
epsilon = 0.1
sigma   = 0.1

def mse(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true), axis=-1)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true), axis=-1)

def svr_cost(y_true, y_pred):
	return C * K.mean(K.maximum(K.abs(y_true-y_pred)- epsilon, 0.) , axis=-1)
    
def svr_laplace_zero_cost(y_true, y_pred):
	return C * K.mean(K.abs(y_true-y_pred) / sigma , axis=-1)


batch_size = 32
epochs     = 100
R          = 10
X = loadmat("/media/Datos/dataset/reg/reg_datasets/bank.mat")

Xr = X['X_tr'] 
Tr = X['T_tr'].T[0]
Xs = X['X_tst']
Ts = X['T_tst'].T[0]

error_hist = []
err        = []
for ii in range(R):
	a     = Input(shape=(Xr.shape[1],))
	b     = Dense(10, activation='relu',W_regularizer=l2(0.5))(a)
	b     = Dense(1, activation='linear',W_regularizer=l2(0.5))(b)
	model = Model(input=a, output=b)
	model.compile(optimizer='adam', metrics=['mae'], loss= 'mse') #svr_cost) #svr_laplace_zero_cost) # #svr_cost) #svr_laplace_zero_cost) 'mse') #  
	history = model.fit(Xr, Tr,
          batch_size      = batch_size, 
          nb_epoch          = epochs,
          validation_data = (Xs, Ts),
          verbose=0)
	ys = model.predict(Xs).T[0]
	error_hist.append(history.history['val_mean_absolute_error'])
	err.append(mae(Ts,ys))

error_hist = np.array(error_hist)
m_err      = np.mean(error_hist,axis=0)
s_err      = np.std(error_hist,axis=0)
err        = np.array(err)
print "mean error : " + str(np.mean(err)) + "  std error : " + str(np.std(err))
plot(m_err)
plot(m_err+s_err,'*-')
plot(m_err-s_err,'*-')
show()
#model.fit(Xr, Tr, batch_size=32, epochs=10, verbose=1, callbacks=None, 
#	validation_split=0.0, validation_data=None, shuffle=True, 
#	class_weight=None, sample_weight=None, initial_epoch=0)