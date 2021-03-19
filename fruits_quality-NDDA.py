# processLBPscores.sh
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


base_dir = "datasets/NDDA/"                                                         # directory path of the apple dataset, download from: https://github.com/OlafenwaMoses/AppleDetection/releases/download/v1/apple_detection_dataset.zip
CATEGORIES = ["Defect","NonDefect"]                                                 # the classes are two: defect and nondefect (good apples)
class_names = CATEGORIES
num_classes = 2
img_size = 100

# Read training set
train_images = []                                                                   # set the training directory in the path


for category in CATEGORIES:                                                         # iterate to each category
    path = os.path.join(base_dir, category)
    class_num = CATEGORIES.index(category)
    for image in os.listdir(path):                                                  # iterate to each image in the category
        if(image.endswith('jpg')):
            img_array = cv2.imread(os.path.join(path,image), cv2.IMREAD_GRAYSCALE)  # read the image
            img_array = cv2.resize(img_array, (img_size, img_size))
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
x_train = x_train.reshape(-1, img_size, img_size, 1)
mean_train = np.mean(x_train, axis=0)
x_train = x_train-mean_train
x_train = x_train/255

# convert label to categorical
y_train = to_categorical(y_train, num_classes)

(x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

print(x_train.shape)
print(x_test.shape)

# Parameters settings
filters_numbers = [8, 16, 32]
filters_size = [[5,5],[4,4],[3,3]]
pool_size=(2, 2)
weight_decay = 1e-5#5e-4#
dropout = 0.5
lr = 0.001
momentum = 0.9
input_shape = x_train.shape[1:]

# Hyperparameters
epochs = 20
batch_size = 32

# Create data generator for data augmentation
train_datagen = ImageDataGenerator(rotation_range=0.1, 
                                width_shift_range=0.1, 
                                height_shift_range=0.1, 
                                zoom_range=0.1, 
                                channel_shift_range=0.05,
                                horizontal_flip=True, 
                                vertical_flip=True,
                                validation_split=0.1)

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size, subset='training') # set as training data
validation_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation') # set as validation data

test_datagen = ImageDataGenerator(rotation_range=0.1,
                                width_shift_range=0.1, 
                                height_shift_range=0.1, 
                                zoom_range=0.1,
                                channel_shift_range=0.05,
                                horizontal_flip=True, 
                                vertical_flip=True)

# build the model
L2_norm = keras.regularizers.l2(weight_decay)

# Setup model layers

# Input layer
model_input = Input(shape=input_shape)

# 1st Convolutional layer
model_output = Conv2D(filters_numbers[0], kernel_size=(filters_size[0]), kernel_regularizer=L2_norm, padding="Same", 
							activation='relu', data_format='channels_last')(model_input)

model_output = BatchNormalization()(model_output)

model_output = MaxPooling2D(pool_size=(pool_size))(model_output)

# 2nd Convolutional layer
model_output = Conv2D(filters_numbers[1], kernel_size=(filters_size[1]), kernel_regularizer=L2_norm, padding="Same",  
							activation='relu', data_format='channels_last')(model_output)

model_output = BatchNormalization()(model_output)

model_output = MaxPooling2D(pool_size=(pool_size))(model_output)

# 3rd Convolutional layer
model_output = Conv2D(filters_numbers[2], kernel_size=(filters_size[2]), kernel_regularizer=L2_norm, padding="Same",  
							activation='relu', data_format='channels_last')(model_output)

model_output = BatchNormalization()(model_output)

model_ouput = GlobalAveragePooling2D(data_format='channels_last')(model_output)

# Convert features to flatten vector      
model_output = Flatten()(model_output)

# Full-connected layer
model_output = Dense(1024)(model_output)
model_output = Dropout(dropout)(model_output)

model_output = Dense(512)(model_output)
model_output = Dropout(dropout)(model_output)

# Output layer
model_output = Dense(num_classes, activation='softmax', name='id')(model_output)

# Create the Model by using Input and Output layers
model = Model(inputs=model_input, outputs=model_output)

# Show the Model summary information
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr, momentum), metrics=['accuracy'])

# Train the model
print("[INFO] Train the model on training data")

history= model.fit(train_generator,
                        validation_data = validation_generator,
                        epochs=epochs, verbose=2)

#model.save('saved_models/fruits_quality-NDDA')

# Plot training curves
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
pyplot.plot(history.history['accuracy'])
pyplot.plot(history.history['val_accuracy'])
pyplot.title('Model Accuracy')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.xticks(x)
pyplot.legend(['Train', 'Validation'], loc='lower right')
pyplot.grid(b=None, which='major', axis='both')
pyplot.savefig('fruits_quality-NDDA-training_acc.png')
pyplot.show()


pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('Model Loss')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.xticks(x)
pyplot.legend(['Train', 'Validation'], loc='upper right')
pyplot.grid(b=None, which='major', axis='both')
pyplot.savefig('fruits_quality-NDDA-training_loss.png')
pyplot.show()

# Test the model
print("[INFO] Evaluate the test data")

results = model.evaluate(test_datagen.flow(x_test, y_test, batch_size=batch_size) , verbose=2)

print('Testing Loss, Testing Acc:', results)

pred = model.predict(x_test)
y_pred = np.argmax(pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print('Accuracy', accuracy_score(y_test, y_pred))
print('Classification report', classification_report(y_test, y_pred, target_names=class_names))


# Plot Testing confusion matrix
mat = confusion_matrix(y_test, y_pred)

figure = plot_confusion_matrix(conf_mat=mat, class_names=class_names, show_absolute=False,
                                show_normed=True, colorbar=True)

pyplot.savefig('fruits_quality-NDDA-confusion-matrix.png')
pyplot.show()

