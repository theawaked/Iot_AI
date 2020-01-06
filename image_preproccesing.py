# dataset opnieuw aanmaken.

import os
import numpy as np
import pandas as pd 
import random
import cv2
import matplotlib.pyplot as plt

from PIL import Image
#%matplotlib inline

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

#assert tf.test.is_gpu_available()


from sklearn.metrics import accuracy_score, confusion_matrix

#K.set_image_data_format('channels_first')
seed = 232
np.random.seed(seed)
tf.random.set_seed(seed)


fig, ax = plt.subplots(2, 3, figsize=(15, 7))
ax = ax.ravel()

plt.tight_layout()

input_path = '../chest-xray-pneumonia//chest_xray/'

def dataset_check():
    for i, _set in enumerate(['train', 'val', 'test']):
        set_path = input_path+_set
        ax[i].imshow(plt.imread(set_path+'/NORMAL/'+os.listdir(set_path+'/NORMAL')[0]), cmap='gray')
        ax[i].set_title('Set: {}, Condition: Normal'.format(_set))
        ax[i+3].imshow(plt.imread(set_path+'/PNEUMONIA/'+os.listdir(set_path+'/PNEUMONIA')[0]), cmap='gray')
        ax[i+3].set_title('Set: {}, Condition: Pneumonia'.format(_set))
    
        
        #dividing the data set in three sets:
        n_normal = len(os.listdir(input_path + _set + '/NORMAL'))
        n_infect = len(os.listdir(input_path + _set + '/PNEUMONIA'))
    print('Set: {}, normal images: {}, pneumonia images: {}'.format(_set, n_normal, n_infect))
    plt.show(block=True)

#dataset_check()

def process_data(img_dims, batch_size):
    # Data generation objects
    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, vertical_flip=True)
    test_val_datagen = ImageDataGenerator(rescale=1./255)
    
    # This is fed to the network in the specified batch sizes and image dimensions
    train_gen = train_datagen.flow_from_directory(
    directory=input_path+'train', 
    target_size=(img_dims, img_dims), 
    batch_size=batch_size, 
    class_mode='binary', 
    shuffle=True)

    validation_gen = test_val_datagen.flow_from_directory(
    directory=input_path+'test', 
    target_size=(img_dims, img_dims), 
    batch_size=batch_size, 
    class_mode='binary', 
    shuffle=True)

    return train_gen, validation_gen, 




img_dims = 150
epochs = 10
batch_size = 32

train_gen, validation_gen = process_data(img_dims, batch_size)

#Convelutional nearal network:
if K.image_data_format() == 'channels_first':
    inputs= Input(shape =(3, img_dims, img_dims))
else:
    inputs = Input(shape=(img_dims, img_dims, 3))

x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
x = MaxPool2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)
x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.2)(x)
#flattens input
x = Flatten()(x)
#densifies the input.
x = Dense(units=128, activation='relu')(x)
x = Dense(units=64, activation='relu')(x)
#reduces overfitting
x = Dropout(rate=0.2)(x)

# Output layer
output = Dense(units=1, activation='sigmoid')(x)

# Creating model and compiling
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', 'accuracy'])

# Callbacks
#saves the best model: 
checkpoint = ModelCheckpoint(filepath='first_try.hdf5', save_best_only=True, save_weights_only=False)
#reduces learning rate: factor = new learnrate * factor 
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')

hist = model.fit_generator(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        epochs=epochs,
        validation_data=validation_gen,
        validation_steps=validation_gen.samples // batch_size, callbacks=[checkpoint, lr_reduce])

model.save_weights('first_try.h5')  # always save your weights after training or during training

#loss and accuracy plot

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(hist.history['accuracy'], label = 'accuracy')
axs[0, 0].plot(hist.history['val_accuracy'], label = 'validation_accuracy')
axs[0, 0].set_title('Model accuracy')
axs[0, 0].legend(['Train', 'Val'], loc='upper left')

axs[0, 1].plot(hist.history['loss'], label = 'loss')
axs[0, 1].plot(hist.history['val_loss'], label = 'val_loss')
axs[0, 1].set_title('Model_loss')
axs[0, 0].legend(['Train', 'Val'], loc='upper left')

axs[1, 0].plot(hist.history['accuracy'], label = 'accuracy')
axs[1, 0].plot(hist.history['loss'])
axs[1, 0].set_title('Train loss and acc')
axs[1, 0].legend(['acc', 'loss'], loc='upper left')

axs[1, 1].plot(hist.history['val_accuracy'], label = 'validation_accuracy')
axs[1, 1].plot(hist.history['val_loss'], label = 'val_loss')
axs[1, 1].set_title('Val loss and acc')
axs[1, 1].legend(['acc', 'loss'], loc='upper left')

try: 
    plt.show()
except:
    print('fout1')


  