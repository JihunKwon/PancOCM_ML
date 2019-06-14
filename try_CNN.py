'''
This code analyzes the OCM waves, and calculates weighting factor by gradient descent (ascend) method
'''
from __future__ import print_function
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

import os

# Hyperparameters
batch_size = 32
epochs = 20
fname1 = 'ocm012_undr2_s1r'
#fname1 = 'ocm012_s1r'
fname2 = '.pkl'

#################### Import experiment 1 ####################
with open(fname1 + str(1) + fname2, 'rb') as f:
    ocm0_all_r1, ocm1_all_r1, ocm2_all_r1 = pickle.load(f)


# concatinate Before and After water for each OCM
ocm0_bef_r1 = ocm0_all_r1[:,:,0]
ocm0_aft_r1 = ocm0_all_r1[:,:,1]
ocm0_r10m_r1 = ocm0_all_r1[:,:,2]

ocm1_bef_r1 = ocm1_all_r1[:,:,0]
ocm1_aft_r1 = ocm1_all_r1[:,:,1]
ocm1_r10m_r1 = ocm1_all_r1[:,:,2]

ocm2_bef_r1 = ocm2_all_r1[:,:,0]
ocm2_aft_r1 = ocm2_all_r1[:,:,1]
ocm2_r10m_r1 = ocm2_all_r1[:,:,2]

# Classify Before and After
ocm0_ba_r1 = np.concatenate([ocm0_bef_r1, ocm0_aft_r1], axis=1)
ocm1_ba_r1 = np.concatenate([ocm1_bef_r1, ocm1_aft_r1], axis=1)
ocm2_ba_r1 = np.concatenate([ocm2_bef_r1, ocm2_aft_r1], axis=1)

# Transpose
ocm0_ba_r1 = ocm0_ba_r1.T
ocm1_ba_r1 = ocm1_ba_r1.T
ocm2_ba_r1 = ocm2_ba_r1.T

# concatinate three OCM sensors
n, t = ocm0_ba_r1.shape
ocm_ba_r1 = np.zeros((n, t ,3))
ocm_b10_r1 = np.zeros((n, t ,3))
ocm_a10_r1 = np.zeros((n, t ,3))

ocm_ba_r1[:,:,0] = ocm0_ba_r1[:,:]
ocm_ba_r1[:,:,1] = ocm1_ba_r1[:,:]
ocm_ba_r1[:,:,2] = ocm2_ba_r1[:,:]
print('ocm_r1 shape:', ocm_ba_r1.shape)


#################### Import experiment 2 ####################
with open(fname1 + str(2) + fname2, 'rb') as f:
    ocm0_all_r2, ocm1_all_r2, ocm2_all_r2 = pickle.load(f)

# concatinate before and after water
ocm0_bef_r2 = ocm0_all_r2[:,:,0]
ocm0_aft_r2 = ocm0_all_r2[:,:,1]
ocm0_r10m_r2 = ocm0_all_r2[:,:,2]

ocm1_bef_r2 = ocm1_all_r2[:,:,0]
ocm1_aft_r2 = ocm1_all_r2[:,:,1]
ocm1_r10m_r2 = ocm1_all_r2[:,:,2]

ocm2_bef_r2 = ocm2_all_r2[:,:,0]
ocm2_aft_r2 = ocm2_all_r2[:,:,1]
ocm2_r10m_r2 = ocm2_all_r2[:,:,2]

# Classify Before and After
ocm0_ba_r2 = np.concatenate([ocm0_bef_r2, ocm0_aft_r2], axis=1)
ocm1_ba_r2 = np.concatenate([ocm1_bef_r2, ocm1_aft_r2], axis=1)
ocm2_ba_r2 = np.concatenate([ocm2_bef_r2, ocm2_aft_r2], axis=1)

# Transpose
ocm0_ba_r2 = ocm0_ba_r2.T
ocm1_ba_r2 = ocm1_ba_r2.T
ocm2_ba_r2 = ocm2_ba_r2.T

# concatinate three OCM sensors
n, t = ocm0_ba_r2.shape
ocm_ba_r2 = np.zeros((n, t ,3))
ocm_b10_r2 = np.zeros((n, t ,3))
ocm_a10_r2 = np.zeros((n, t ,3))

ocm_ba_r2[:,:,0] = ocm0_ba_r2[:,:]
ocm_ba_r2[:,:,1] = ocm1_ba_r2[:,:]
ocm_ba_r2[:,:,2] = ocm2_ba_r2[:,:]

print('ocm_r2 shape:', ocm_ba_r2.shape)

#################### Pre Proccesing ####################
# Calculate mean and diviation
ocm_ba_r12 = np.concatenate([ocm0_ba_r1, ocm0_ba_r2, ocm1_ba_r1, ocm1_ba_r2, ocm2_ba_r1, ocm2_ba_r2], axis=0)
ocm_ba_m = np.mean(ocm_ba_r12)
ocm_ba_v = np.var(ocm_ba_r12)

# Standardization
#ocm_ba_r1 = (ocm_ba_r1 - ocm_ba_m) / ocm_ba_v
#ocm_ba_r2 = (ocm_ba_r2 - ocm_ba_m) / ocm_ba_v

# Create Answer
y_ba_r1 = np.zeros(ocm_ba_r1.shape[0])
y_ba_r1[ocm0_bef_r1.shape[1]:] = 1
y_ba_r2 = np.zeros(ocm_ba_r2.shape[0])
y_ba_r2[ocm0_bef_r2.shape[1]:] = 1

###################### Start Keras ##########################
# The data, split between train and test sets:
#X_train, X_test, y_train, y_test = train_test_split(ocm_ba_r1, y_ba_r1, test_size=0.5, random_state=1)

X_train = ocm_ba_r1
X_test = ocm_ba_r2
y_train = y_ba_r1.astype(int)
y_test = y_ba_r2.astype(int)

# Build NN
model = Sequential()
model.add(Conv1D(16, 8, padding='same', input_shape=X_train.shape[1:], activation='relu'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Dropout(0.5))

model.add(Conv1D(8, 8, padding='same', activation='relu'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

# initiate RMSprop optimizer
opt = keras.optimizers.adam(lr=0.00001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    shuffle=True,
                    callbacks=[es])

# Score trained model.
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# ----------------------------------------------
# Some plots
# ----------------------------------------------
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# loss
def plot_history_loss(fit):
    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    axL.plot(fit.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# acc
def plot_history_acc(fit):
    # Plot the loss in the history
    axR.plot(fit.history['acc'],label="accuracy for training")
    axR.plot(fit.history['val_acc'],label="accuracy for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='lower right')

plot_history_loss(history)
plot_history_acc(history)
fig.savefig('./result.png')
plt.close()


### ROC ###
from sklearn.metrics import roc_curve, auc

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)
y_pred_class = model.predict_classes(X_test)
fpr, tpr, thr = roc_curve(y_test, y_pred)
auc_keras = auc(fpr, tpr)

fig2 = plt.subplots(ncols=1, figsize=(5,4))
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('./ROC.png')