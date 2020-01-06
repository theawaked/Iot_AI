import numpy as np
import os
from keras.models import load_model
from PIL import Image
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

img_dims =150
#Redefine model to load the weights:
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
checkpoint = ModelCheckpoint(filepath='first_try.hdf5', save_best_only=True, save_weights_only=False)
# lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
# early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')


model.load_weights('first_try.hdf5')
#image inlezen.
imagepath = 'C:/Users/gerben/Desktop/school/jaar 3/AI/chest-xray-pneumonia/chest_xray/val/both/'



def test_picture(fileandpicturename):
          filepath = str(fileandpicturename)
          image1 = image.load_img(imagepath+filepath,target_size=(img_dims, img_dims))
          img_tensor = image.img_to_array(image1)                    
          img_tensor = np.expand_dims(img_tensor, axis=0)       
          img_tensor /= 255.                                      
         
         
          probabilities = model.predict(img_tensor)
          print(probabilities)


for img in (os.listdir(imagepath)):
    print(img)
    test_picture(img)
        

