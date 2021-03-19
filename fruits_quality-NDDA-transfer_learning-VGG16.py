# fruits_quality-NDDA-transfer_learning-VGG16.py
#
# Copyright 2020 by Ruber Hernández-García (linkedin.com/in/ruberhg) and LITRP (www.litrp.cl)
# All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License GPLv3 as published by
# the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  Please see the
# GNU General Public License for more details: https://www.gnu.org/licenses/gpl-3.0.html

# __author__ = "Ruber Hernández-García"
# __copyright__ = "Copyright (C) 2020 Ruber Hernández-García"
# __license__ = "GNU General Public License GPLv3"
# __version__ = "1.0"

import os 																			# for manipulating the directories
import cv2 																			# for image processing 
import random 																		# for shuffling
import numpy as np 																	# for array manipulating and scientific computing
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import tensorflow as tf 															# for more details see: https://www.tensorflow.org/tutorials
from tensorflow import keras 														# for more details see: https://www.tensorflow.org/guide/keras/overview

from tensorflow.keras.models import Model 											# for more details see about Model class API: https://keras.io/models/model/
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Flatten, BatchNormalization              # for model layers
from tensorflow.keras.utils import to_categorical									# for categorical labels
from tensorflow.keras import optimizers                                             # for model optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator                 # for data augmentation, more detail: https://keras.io/preprocessing/image/
from tensorflow.keras.applications import VGG16                                     # load VGG16 library
from tensorflow.keras.applications.resnet50 import preprocess_input



base_dir = "datasets/NDDA/"                                                         # directory path of the apple dataset, download from: https://github.com/OlafenwaMoses/AppleDetection/releases/download/v1/apple_detection_dataset.zip
CATEGORIES = ["Defect","NonDefect"]                                                 # the classes are two: defect and nondefect (good apples)
num_classes = 2
img_size = 100
batch_size = 16

# Read training set
train_images = []                                                                   # set the training directory in the path


for category in CATEGORIES:                                                         # iterate to each category
    path = os.path.join(base_dir, category)
    class_num = CATEGORIES.index(category)
    for image in os.listdir(path):                                                  # iterate to each image in the category
        if(image.endswith('jpg')):
            img_array = cv2.imread(os.path.join(path,image))                        # read the image
            img_array = cv2.resize(img_array, (img_size, img_size))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_array = np.expand_dims(img_array, axis=2)
            train_images.append([img_array, class_num])                             # save the image in train data array

print("Images data: ", len(train_images))                                           

# Shuffle the dataset before training for better accuracy
x_train = []																		# array for images
y_train = []																		# array for labels

random.shuffle(train_images)														# shuffle training images

for features, label in train_images: 												# iterate to each image and the corresponding label in training data
	x_train.append(features)
	y_train.append(label)
x_train = np.array(x_train)

# reshape and normalize the data before training
x_train = x_train.reshape(-1, img_size, img_size, 3)
# mean_train = np.mean(x_train, axis=0)
# x_train = x_train-mean_train
# x_train = x_train/255
x_train = preprocess_input(x_train)

# convert label to categorical
y_train = to_categorical(y_train, num_classes)

(x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

print(x_train.shape)
print(x_test.shape)

# Create data generator for data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=10, 
    width_shift_range=0.1,
    height_shift_range=0.1, 
    zoom_range=0.1,
    horizontal_flip=True, 
    vertical_flip=True,
    validation_split=0.1)

# Generate training and validation data using ImageDataGenerator
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size, subset='training') # set as training data
validation_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation') # set as validation data


# Parameters settings
lr = 1e-3
momentum = 0.9

# Hyperparameters
epochs = 10


# Using the pre-trained model: 

# Import the pre-trained model VGG16 and use the weights from ImageNet
base_model = VGG16(
    weights='imagenet',                     # Load pre-trained weights on ImageNet
    input_shape=(img_size,img_size,3),      # Lower down the shape for the input images to (100, 100, 3)
    include_top=False)                      # Do not include the ImageNet classifier at the top.
base_model.trainable = False                # Freeze all layers in the base model

# Create a new model on top of the output of one (or several) layers from the base model

# Adding new FC Layers to the modified model
modified_VGG16_output = base_model.output
modified_VGG16_output = GlobalAveragePooling2D()(modified_VGG16_output)
modified_VGG16_output = Dense(512, activation='relu')(modified_VGG16_output)
modified_VGG16_output = Dense(num_classes, activation='softmax')(modified_VGG16_output)

# Create the new model by using Input and Output layers
new_VGG16_model = Model(inputs=base_model.input, outputs=modified_VGG16_output)

new_VGG16_model.summary()


# Compile the model with categorical crossentropy loss function and SGD optimizer
new_VGG16_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr, momentum), metrics=['accuracy'])


# Train the model
print("[INFO] Train the model on training data")

history= new_VGG16_model.fit(train_generator,
                        validation_data = validation_generator,
                        epochs=epochs, verbose=1)

#new_VGG16_model.save('VGG16_TransferLearning_NDDA.h5')

# Plot training curves
x = [1,2,3,4,5,6,7,8,9,10]
pyplot.plot(history.history['accuracy'])
pyplot.plot(history.history['val_accuracy'])
pyplot.title('Model Accuracy')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.xticks(x)
pyplot.legend(['Train', 'Validation'], loc='lower right')
pyplot.grid(b=None, which='major', axis='both')

pyplot.savefig('fruits_quality-NDDA-VGG16-training_acc-4.png')
#pyplot.show()


pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('Model Loss')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.xticks(x)
pyplot.legend(['Train', 'Validation'], loc='upper right')
pyplot.grid(b=None, which='major', axis='both')

pyplot.savefig('fruits_quality-NDDA-VGG16-training_loss-4.png')
#pyplot.show()

# Test the model
print("[INFO] Evaluate the test data")

results = new_VGG16_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

print('Testing Loss, Testing Acc:', results)

pred = new_VGG16_model.predict(x_test)
y_pred = np.argmax(pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print('Accuracy', accuracy_score(y_test, y_pred))
print('Classification report', classification_report(y_test, y_pred, target_names=CATEGORIES))


# Plot Testing confusion matrix
mat = confusion_matrix(y_test, y_pred)

figure = plot_confusion_matrix(conf_mat=mat, class_names=CATEGORIES, show_absolute=False,
                                show_normed=True, colorbar=True)

pyplot.savefig('fruits_quality-NDDA-VGG16-confusion-matrix-4.png')
#pyplot.show()

