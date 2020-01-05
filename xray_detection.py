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


# First conv block
x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
x = MaxPool2D(pool_size=(2, 2))(x)
# Second conv block
#x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)

#x = BatchNormalization()(x)
#x = MaxPool2D(pool_size=(2, 2))(x)

# Third conv block
#x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
#x = BatchNormalization()(x)
#x = MaxPool2D(pool_size=(2, 2))(x)

# Fourth conv block
#x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
#x = BatchNormalization()(x)
#x = MaxPool2D(pool_size=(2, 2))(x)
#x = Dropout(rate=0.2)(x)

# Fifth conv block
#x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
#x = BatchNormalization()(x)
#x = MaxPool2D(pool_size=(2, 2))(x)
#x = Dropout(rate=0.2)(x)

# FC layer
x = Flatten()(x)
x = Dense(units=128, activation='relu')(x)
#x = Dropout(rate=0.7)(x)
#x = Dense(units=128, activation='relu')(x)
#x = Dropout(rate=0.5)(x)
#x = Dense(units=64, activation='relu')(x)
#x = Dropout(rate=0.3)(x)

# Output layer
output = Dense(units=1, activation='sigmoid')(x)

# Creating model and compiling
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy','categorical_accuracy', 'accuracy'])

 # Callbacks
checkpoint = ModelCheckpoint(filepath='first_try.hdf5', save_best_only=True, save_weights_only=False)
# lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
# early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')


model.load_weights('first_try.hdf5')
#image inlezen.
imagepath = 'C:/Users/gerben/Desktop/school/jaar 3/AI/chest-xray-pneumonia/chest_xray/val/'



def test_picture(fileandpicturename):
          filepath = str(fileandpicturename)
          image1 = image.load_img(imagepath+filepath,target_size=(img_dims, img_dims))
          img_tensor = image.img_to_array(image1)                    
          img_tensor = np.expand_dims(img_tensor, axis=0)       
          img_tensor /= 255.                                      
         
         
          probabilities = model.predict(img_tensor)
          print(probabilities)

# test_data_dir = imagepath + 'both'
# test_datagen = ImageDataGenerator(
#     rescale=1. / 255)

# test_generator = test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(img_dims, img_dims),
#     batch_size= 4,
# 	shuffle='False',
#     class_mode='categorical')


# test_generator.reset()
# Y_pred = model.predict_generator(test_generator)
# classes = test_generator.classes[test_generator.index_array]
# y_pred = np.argmax(Y_pred, axis=-1)
# sum(y_pred==classes)/10000

# 0.9922

# from sklearn.metrics import confusion_matrix
# confusion_matrix(test_generator.classes[test_generator.index_array],y_pred)

for cond in ['/NORMAL/', '/PNEUMONIA/']:
    print(cond)
    for img in (os.listdir(imagepath + cond)):
        print(img)
        test_picture(cond + img)
        

