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
import os

# Hyperparameters
batch_size = 32
epochs = 5

############# Import experiment 1 #############
with open('ocm012_s1r1_1000.pkl', 'rb') as f:
    ocm0_all_1, ocm1_all_1, ocm2_all_1 = pickle.load(f)

print('ocm0 shape', ocm0_all_1.shape)
print('ocm1 shape', ocm1_all_1.shape)
print('ocm2 shape', ocm2_all_1.shape)

# concatinate before and after water
ocm0_bef_1 = ocm0_all_1[:,:,0]
ocm0_aft_1 = ocm0_all_1[:,:,1]
ocm0_10m_1 = ocm0_all_1[:,:,2]
ocm0_ba_1 = np.concatenate([ocm0_bef_1, ocm0_aft_1], axis=1)
print(np.sum(ocm0_bef_1))
print(np.sum(ocm0_aft_1))
print(np.sum(ocm0_10m_1))

# Transpose
ocm0_1 = ocm0_ba_1.T

# concatinate three OCM sensors
n, t = ocm0_1.shape
ocm_1 = np.zeros((n, t ,1))
ocm_1[:,:,0] = ocm0_1[:,:]
print('ocm_1 shape:', ocm_1.shape)

# Calculate mean and diviation
ocm_m = np.mean(ocm_1)
ocm_v = np.var(ocm_1)

# Standardization
ocm_1 = (ocm_1 - ocm_m) / ocm_v

# Create Answer
y_1 = np.zeros(ocm0_1.shape[0])
y_1[ocm0_bef_1.shape[1]:ocm0_bef_1.shape[1]*2] = 1
#y_1[ocm0_bef_1.shape[0]*2:] = 2
print(y_1.shape)

############# Import experiment 2 #############
with open('ocm012_s1r2_1000.pkl', 'rb') as f:
    ocm0_all_2, ocm1_all_2, ocm2_all_2 = pickle.load(f)

# concatinate before and after water
ocm0_bef_2 = ocm0_all_2[:,:,0]
ocm0_aft_2 = ocm0_all_2[:,:,1]
ocm0_10m_2 = ocm0_all_2[:,:,2]
ocm0_ba_2 = np.concatenate([ocm0_bef_2, ocm0_aft_2], axis=1)

# Transpose
ocm0_2 = ocm0_ba_2.T

# concatinate three OCM sensors
n, t = ocm0_2.shape
ocm_2 = np.zeros((n, t ,1))
ocm_2[:,:,0] = ocm0_2[:,:]
print('ocm_2 shape:', ocm_2.shape)

# Calculate mean and diviation
ocm_m = np.mean(ocm_2)
ocm_v = np.var(ocm_2)

# Standardization
ocm_2 = (ocm_2 - ocm_m) / ocm_v

# Create Answer
y_2 = np.zeros(ocm0_2.shape[0])
y_2[ocm0_bef_2.shape[1]:ocm0_bef_2.shape[1]*2] = 1
#y_2[ocm0_bef_2.shape[0]*2:] = 2

###################### Start Keras ##########################
# The data, split between train and test sets:
print('ocm:', ocm_1.shape)
print('y:', y_1.shape)
#X_train, X_test, y_train, y_test = train_test_split(ocm_1, y, test_size=0.33, random_state=1)
X_train = ocm_1
X_test = ocm_2
y_train = y_1.astype(int)
y_test = y_2.astype(int)

print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

# Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, 2)
#y_test = keras.utils.to_categorical(y_test, 2)

#y = y_test.ravel()

# Build NN
model = Sequential()

model.add(Conv1D(8, 4, padding='same', input_shape=X_train.shape[1:], activation='relu'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print('x_train shape_2:', X_train.shape)
print('x_test shape_2:', X_test.shape)
print('y_train shape_2:', y_train.shape)
print('y_test shape_2:', y_test.shape)

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test)
          ,shuffle=True)


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
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import auc


### June 11 ###
'''
print('X_test shape',X_test.shape)
y_pred_keras = model.predict(X_test).ravel()
print('y_pred_keras shape',y_pred_keras.shape)
print('y_pred_keras: ',y_pred_keras)
print('y_test shape',y_test.shape)
#y_pred_keras = model.predict(X_test).ravel()
y = y_test.flatten()
print('y: ',y)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
print('here')
auc_keras = auc(fpr_keras, tpr_keras)
'''
### June 12 ###
y_pred = model.predict(X_test)
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
'''
# Zoom in view of the upper left corner.
fig3 = plt.subplots(ncols=1, figsize=(5,4))
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.savefig('./ROC_zoom.png')

'''