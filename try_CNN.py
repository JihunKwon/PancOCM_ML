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
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
import os

# Hyperparameters
batch_size = 32
num_classes = 2
epochs = 1

with open('t012_fidx12.pkl', 'rb') as f:
    ocm0_all, ocm1_all, ocm2_all = pickle.load(f)

# concatinate before and after water
ocm0_bef = ocm0_all[:,:,0]
ocm0_aft = ocm0_all[:,:,1]
ocm0 = np.concatenate([ocm0_bef, ocm0_aft], axis=1)

ocm1_bef = ocm1_all[:,:,0]
ocm1_aft = ocm1_all[:,:,1]
ocm1 = np.concatenate([ocm1_bef, ocm1_aft], axis=1)

ocm2_bef = ocm2_all[:,:,0]
ocm2_aft = ocm2_all[:,:,1]
ocm2 = np.concatenate([ocm2_bef, ocm2_aft], axis=1)

# Transpose
ocm0 = ocm0.T
ocm1 = ocm1.T
ocm2 = ocm2.T

# concatinate three OCM sensors
print('ocm0 shape:', ocm0.shape)
n, t = ocm0.shape
ocm = np.zeros((n, t ,3))
ocm[:,:,0] = ocm0[:,:]
ocm[:,:,1] = ocm1[:,:]
ocm[:,:,2] = ocm2[:,:]
#ocm = np.concatenate([ocm0, ocm1, ocm2], axis=2)
print('ocm shape:', ocm.shape)

# Calculate mean and diviation
ocm_m = np.mean(ocm)
ocm_v = np.var(ocm)

# Standardization
ocm = (ocm - ocm_m) / ocm_v

# Create Answer
y = np.zeros(ocm0.shape[0])
y[ocm0_bef.shape[0]:] = 1


###################### Start Keras ##########################
# The data, split between train and test sets:
print('ocm:',ocm.shape)
print('y:',y.shape)
X_train, X_test, y_train, y_test = train_test_split(ocm, y, test_size=0.33, random_state=1)
print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build NN
model = Sequential()
model.add(Conv1D(32, 5, padding='same', input_shape=X_train.shape[1:], activation='relu'))
model.add(Conv1D(32, 5, padding='same', activation='relu'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Dropout(0.25))

model.add(Conv1D(64, 5, padding='same', activation='relu'))
model.add(Conv1D(64, 5, padding='same', activation='relu'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Dropout(0.25))

model.add(Conv1D(128, 5, padding='same', activation='relu'))
model.add(Conv1D(128, 5, padding='same', activation='relu'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          shuffle=True)


# Score trained model.
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()