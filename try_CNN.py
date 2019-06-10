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
batch_size = 540
num_classes = 3
epochs = 50

############# Import experiment 1 #############
with open('ocm012_s1r1.pkl', 'rb') as f:
    ocm0_all_1, ocm1_all_1, ocm2_all_1 = pickle.load(f)

# concatinate before and after water
ocm0_bef_1 = ocm0_all_1[:,:,0]
ocm0_aft_1 = ocm0_all_1[:,:,1]
ocm0_10m_1 = ocm0_all_1[:,:,2]
ocm0_1 = np.concatenate([ocm0_bef_1, ocm0_aft_1, ocm0_10m_1], axis=1)

ocm1_bef_1 = ocm1_all_1[:,:,0]
ocm1_aft_1 = ocm1_all_1[:,:,1]
ocm1_10m_1 = ocm1_all_1[:,:,2]
ocm1_1 = np.concatenate([ocm1_bef_1, ocm1_aft_1, ocm1_10m_1], axis=1)

ocm2_bef_1 = ocm2_all_1[:,:,0]
ocm2_aft_1 = ocm2_all_1[:,:,1]
ocm2_10m_1 = ocm2_all_1[:,:,2]
ocm2_1 = np.concatenate([ocm2_bef_1, ocm2_aft_1, ocm2_10m_1], axis=1)

# Transpose
ocm0_1 = ocm0_1.T
ocm1_1 = ocm1_1.T
ocm2_1 = ocm2_1.T

# concatinate three OCM sensors
n, t = ocm0_1.shape
ocm_1 = np.zeros((n, t ,3))
ocm_1[:,:,0] = ocm0_1[:,:]
ocm_1[:,:,1] = ocm1_1[:,:]
ocm_1[:,:,2] = ocm2_1[:,:]
#ocm = np.concatenate([ocm0, ocm1, ocm2], axis=2)
print('ocm_1 shape:', ocm_1.shape)

# Calculate mean and diviation
ocm_m = np.mean(ocm_1)
ocm_v = np.var(ocm_1)

# Standardization
ocm_1 = (ocm_1 - ocm_m) / ocm_v

# Create Answer
y_1 = np.zeros(ocm0_1.shape[0])
y_1[ocm0_bef_1.shape[0]:ocm0_bef_1.shape[0]*2] = 1
y_1[ocm0_bef_1.shape[0]*2:] = 2


############# Import experiment 2 #############
with open('ocm012_s1r2.pkl', 'rb') as f:
    ocm0_all_2, ocm1_all_2, ocm2_all_2 = pickle.load(f)

# concatinate before and after water
ocm0_bef_2 = ocm0_all_2[:,:,0]
ocm0_aft_2 = ocm0_all_2[:,:,1]
ocm0_10m_2 = ocm0_all_2[:,:,2]
ocm0_2 = np.concatenate([ocm0_bef_2, ocm0_aft_2, ocm0_10m_2], axis=1)

ocm1_bef_2 = ocm1_all_2[:,:,0]
ocm1_aft_2 = ocm1_all_2[:,:,1]
ocm1_10m_2 = ocm1_all_2[:,:,2]
ocm1_2 = np.concatenate([ocm1_bef_2, ocm1_aft_2, ocm1_10m_2], axis=1)

ocm2_bef_2 = ocm2_all_2[:,:,0]
ocm2_aft_2 = ocm2_all_2[:,:,1]
ocm2_10m_2 = ocm2_all_2[:,:,2]
ocm2_2 = np.concatenate([ocm2_bef_2, ocm2_aft_2, ocm2_10m_2], axis=1)

# Transpose
ocm0_2 = ocm0_2.T
ocm1_2 = ocm1_2.T
ocm2_2 = ocm2_2.T

# concatinate three OCM sensors
n, t = ocm0_2.shape
ocm_2 = np.zeros((n, t ,3))
ocm_2[:,:,0] = ocm0_2[:,:]
ocm_2[:,:,1] = ocm1_2[:,:]
ocm_2[:,:,2] = ocm2_2[:,:]
#ocm = np.concatenate([ocm0, ocm1, ocm2], axis=2)
print('ocm_2 shape:', ocm_2.shape)

# Calculate mean and diviation
ocm_m = np.mean(ocm_2)
ocm_v = np.var(ocm_2)

# Standardization
ocm_2 = (ocm_2 - ocm_m) / ocm_v

# Create Answer
y_2 = np.zeros(ocm0_2.shape[0])
y_2[ocm0_bef_2.shape[0]:ocm0_bef_2.shape[0]*2] = 1
y_2[ocm0_bef_2.shape[0]*2:] = 2

###################### Start Keras ##########################
# The data, split between train and test sets:
print('ocm:', ocm_1.shape)
print('y:', y_1.shape)
#X_train, X_test, y_train, y_test = train_test_split(ocm_1, y, test_size=0.33, random_state=1)
X_train = ocm_1
X_test = ocm_2
y_train = y_1
y_test = y_2

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build NN
model = Sequential()
model.add(Conv1D(16, 8, padding='same', input_shape=X_train.shape[1:], activation='relu'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.000001, decay=1e-6)

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