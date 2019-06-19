'''
This code trains model by BreathHold (bh) = 1 and Test performance by bh=2~5
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
from keras.regularizers import l1_l2
import time

start = time.time()

# Hyperparameters
batch_size = 32
epochs = 5
#fname1 = 'ocm012_undr2_s1r'
#fname1 = 'ocm012_s1r'
#fname2 = '.pkl'

#%%%%%%%%%%%%%%%%%%% Import data %%%%%%%%%%%%%%%%%%%
#################### S1r1 ####################
with open('Raw_det_ocm012_undr4_s1r1.pkl', 'rb') as f:
    ocm0_all, ocm1_all, ocm2_all = pickle.load(f)

print(ocm0_all.shape)  # (350, 10245, 3)
bh = ocm0_all.shape[1]//5
# concatinate Before and After water for each OCM
phase = 0  # Before water
ocm0_bef_train = ocm0_all[:,0:bh,phase]
ocm1_bef_train = ocm1_all[:,0:bh,phase]
ocm2_bef_train = ocm2_all[:,0:bh,phase]
ocm0_bef_test = ocm0_all[:,bh:bh*5,phase]
ocm1_bef_test = ocm1_all[:,bh:bh*5,phase]
ocm2_bef_test = ocm2_all[:,bh:bh*5,phase]

phase = 1  # After water
ocm0_aft_train = ocm0_all[:,0:bh,phase]
ocm1_aft_train = ocm1_all[:,0:bh,phase]
ocm2_aft_train = ocm2_all[:,0:bh,phase]
ocm0_aft_test = ocm0_all[:,bh:bh*5,phase]
ocm1_aft_test = ocm1_all[:,bh:bh*5,phase]
ocm2_aft_test = ocm2_all[:,bh:bh*5,phase]

# allocate to one variable
ocm_bef_train = np.zeros((ocm0_bef_train.shape[0], ocm0_bef_train.shape[1], 3))
ocm_aft_train = np.zeros((ocm0_aft_train.shape[0], ocm0_aft_train.shape[1], 3))
ocm_bef_test = np.zeros((ocm0_bef_test.shape[0], ocm0_bef_test.shape[1], 3))
ocm_aft_test = np.zeros((ocm0_aft_test.shape[0], ocm0_aft_test.shape[1], 3))

ocm_bef_train[:,:,0] = ocm0_bef_train[:,:]
ocm_bef_train[:,:,1] = ocm1_bef_train[:,:]
ocm_bef_train[:,:,2] = ocm2_bef_train[:,:]
ocm_bef_test[:,:,0] = ocm0_bef_test[:,:]
ocm_bef_test[:,:,1] = ocm1_bef_test[:,:]
ocm_bef_test[:,:,2] = ocm2_bef_test[:,:]
ocm_aft_train[:,:,0] = ocm0_aft_train[:,:]
ocm_aft_train[:,:,1] = ocm1_aft_train[:,:]
ocm_aft_train[:,:,2] = ocm2_aft_train[:,:]
ocm_aft_test[:,:,0] = ocm0_aft_test[:,:]
ocm_aft_test[:,:,1] = ocm1_aft_test[:,:]
ocm_aft_test[:,:,2] = ocm2_aft_test[:,:]

# Transpose
ocm_bef_train = np.einsum('abc->bac', ocm_bef_train)
ocm_aft_train = np.einsum('abc->bac', ocm_aft_train)
ocm_bef_test = np.einsum('abc->bac', ocm_bef_test)
ocm_aft_test = np.einsum('abc->bac', ocm_aft_test)
print('before train:', ocm_bef_train.shape)
print('after  train:', ocm_bef_train.shape)
print('before test:', ocm_aft_test.shape)
print('after  test:', ocm_aft_test.shape)

#%%%%%%%%%%%%%%%%%%% Pre Proccesing %%%%%%%%%%%%%%%%%%%
# Calculate mean and diviation
ocm_bef_train_m = np.mean(ocm_bef_train)
ocm_aft_train_m = np.mean(ocm_aft_train)
ocm_bef_test_m = np.mean(ocm_bef_test)
ocm_aft_test_m = np.mean(ocm_aft_test)
ocm_bef_train_v = np.var(ocm_bef_train)
ocm_aft_train_v = np.var(ocm_aft_train)
ocm_bef_test_v = np.var(ocm_bef_test)
ocm_aft_test_v = np.var(ocm_aft_test)

# Standardization
ocm_bef_train = (ocm_bef_train - ocm_bef_train_m) / ocm_bef_train_v
ocm_aft_train = (ocm_aft_train - ocm_aft_train_m) / ocm_aft_train_v
ocm_bef_test = (ocm_bef_test - ocm_bef_test_m) / ocm_bef_test_v
ocm_aft_test = (ocm_aft_test - ocm_aft_test_m) / ocm_aft_test_v

# Concatenate Training set
# All subject (except s3r2)
ocm_ba_train = np.zeros((ocm_bef_train.shape[0]+ocm_aft_train.shape[0], ocm_bef_train.shape[1]))
ocm_ba_test = np.zeros((ocm_bef_test.shape[0]+ocm_aft_test.shape[0], ocm_bef_test.shape[1]))
ocm_ba_train = np.concatenate([ocm_bef_train, ocm_aft_train], axis=0)
ocm_ba_test = np.concatenate([ocm_bef_test, ocm_aft_test], axis=0)
print('ocm_ba_train shape:', ocm_ba_train.shape)
print('ocm_ba_test shape:', ocm_ba_test.shape)

# Create Answer
y_ba_train = np.zeros(ocm_ba_train.shape[0])
y_ba_train[ocm_bef_train.shape[1]:] = 1
y_ba_test = np.zeros(ocm_ba_test.shape[0])
y_ba_test[ocm_bef_test.shape[1]:] = 1
print('y_ba_train shape:', y_ba_train.shape)
print('y_ba_test shape:', y_ba_test.shape)

#%%%%%%%%%%%%%%%%%%% Start Keras %%%%%%%%%%%%%%%%%%%
# The data, split between train and test sets:
#X_train, X_test, y_train, y_test = train_test_split(ocm_ba_all_r1, y_ba_all_r1, test_size=0.5, random_state=1)

X_train = ocm_ba_train
X_test = ocm_ba_test
y_train = y_ba_train.astype(int)
y_test = y_ba_test.astype(int)

n_timesteps, n_features = X_train.shape[1], X_train.shape[2]

# Build NN
n_conv = 128
n_dense = 128

model = Sequential()
model.add(Conv1D(filters=n_conv, kernel_size=3, padding='same', input_shape=(n_timesteps, n_features), activation='relu'))
model.add(Conv1D(filters=n_conv, kernel_size=3, activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_size=2, padding='same'))

model.add(Conv1D(filters=n_conv * 2, kernel_size=3, activation='relu'))
model.add(Conv1D(filters=n_conv * 2, kernel_size=3, activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_size=2, padding='same'))

model.add(Conv1D(filters=n_conv * 4, kernel_size=3, activation='relu'))
model.add(Conv1D(filters=n_conv * 4, kernel_size=3, activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_size=2, padding='same'))

model.add(Flatten())
model.add(Dense(n_dense, activation='relu', W_regularizer = l1_l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(int(n_dense * 0.4), activation='relu', W_regularizer = l1_l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# initiate optimizer
opt = keras.optimizers.adam(lr=0.00005, decay=1e-6)

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
                    shuffle=True)#,
                    #callbacks=[es])

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
    axL.legend(loc='best')

# acc
def plot_history_acc(fit):
    # Plot the loss in the history
    axR.plot(fit.history['acc'],label="accuracy for training")
    axR.plot(fit.history['val_acc'],label="accuracy for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='best')

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

print(time.time() - start)
