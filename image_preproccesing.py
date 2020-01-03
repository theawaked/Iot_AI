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

seed = 232
np.random.seed(seed)
tf.random.set_seed(seed)


fig, ax = plt.subplots(2, 3, figsize=(15, 7))
ax = ax.ravel()

plt.tight_layout()

input_path = '../chest-xray-pneumonia//chest_xray/chest_xray/'

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

    test_gen = test_val_datagen.flow_from_directory(
    directory=input_path+'test', 
    target_size=(img_dims, img_dims), 
    batch_size=batch_size, 
    class_mode='binary', 
    shuffle=True)
    
    # I will be making predictions off of the test set in one batch size
    # This is useful to be able to get the confusion matrix
    test_data = []
    test_labels = []

    for cond in ['/NORMAL/', '/PNEUMONIA/']:
        for img in (os.listdir(input_path + 'test' + cond)):
           
            print(img)
            # title = str(img)
            # image2= Image.open(input_path+'test'+cond+img)
            # image2.convert(mode = 'L')
            # image2.save(title)
            #print(imgage2)

            img = cv2.imread(input_path+'test'+cond+img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(img.shape)
            img = cv2.resize(img, (img_dims, img_dims))
            img = np.dstack([img, img, img])
            
            img = img.astype('float32') / 255
            #print(img)
            if cond=='/NORMAL/':
                label = 0
            elif cond=='/PNEUMONIA/':
                label = 1
            test_data.append(img)
            test_labels.append(label)
         
    test_data1 = np.array(test_data)
    test_labels1 = np.array(test_labels)
    

    return train_gen, test_gen, test_data1, test_labels1



img_dims = 150
epochs = 10
batch_size = 32

train_gen, test_gen, test_data, test_labels = process_data(img_dims, batch_size)