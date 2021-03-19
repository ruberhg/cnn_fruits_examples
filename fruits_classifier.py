# fruit_classifier.py
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
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import tensorflow as tf 															# for more details see: https://www.tensorflow.org/tutorials
from tensorflow import keras 														# for more details see: https://www.tensorflow.org/guide/keras/overview

from tensorflow.keras.models import Model 								            # for more details see about Model class API: https://keras.io/models/model/
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical				       				# for categorical labels
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback                  # Tensorboard


NAME = 'fruits-classifier'                                                          # name for the callback output
base_dir = "datasets/fruits-360_dataset/fruits-360/"	                            # directory path of the fruit dataset, download from: https://github.com/Horea94/Fruit-Images-Dataset (Horea Muresan, Mihai Oltean, Fruit recognition from images using deep learning, Acta Univ. Sapientiae, Informatica Vol. 10, Issue 1, pp. 26-42, 2018.)
CATEGORIES = ["Apple Golden 1","Apple Pink Lady","Apple Red 1","Pear Red","Pear Williams","Pear Monster"]			# we work with three classes of Apple and Pear
class_names = CATEGORIES
num_classes = 6
img_size = 100


# Read training set
train_images = []
train_dir = os.path.join(base_dir, 'Training/')										# set the training directory in the path

for category in CATEGORIES:															# iterate to each category
    path = os.path.join(train_dir, category)
    class_num = CATEGORIES.index(category)
    for image in os.listdir(path):													# iterate to each image in the category
        if(image.endswith('jpg')):
            img_array = cv2.imread(os.path.join(path,image), cv2.IMREAD_GRAYSCALE)	# read the image
            train_images.append([img_array, class_num])								# save the image in train data array

print("Training images: ", len(train_images))											

# Read tesing set
test_images = []
test_dir = os.path.join(base_dir, 'Test/')											# set the test directory in the path

for category in CATEGORIES:															# iterate to each category
    path = os.path.join(test_dir, category)
    class_num = CATEGORIES.index(category)
    for image in os.listdir(path):													# iterate to each image in the category
        if(image.endswith('jpg')):													
            img_array = cv2.imread(os.path.join(path,image), cv2.IMREAD_GRAYSCALE)	# read the image
            test_images.append([img_array, class_num])								# save the image in train data array
            
print("Testing images: ", len(test_images))

# Shuffle the dataset before training for better accuracy
x_train = []																		# array for images
y_train = []																		# array for labels

random.shuffle(train_images)														# shuffle training images

for features, label in train_images: 												# iterate to each image and the corresponding label in training data
	x_train.append(features)
	y_train.append(label)
x_train = np.array(x_train)
 
x_test = []																			# array for images
y_test = []																			# array for labels

random.shuffle(test_images) 														# shuffle testing images

for features, label in test_images: 												# iterate to each image and the corresponding label in training data
	x_test.append(features)
	y_test.append(label)
x_test = np.array(x_test)

# reshape and normalize the data before training
x_train = x_train.reshape(-1, img_size, img_size, 1)
mean_train = np.mean(x_train, axis=0)
x_train = x_train-mean_train
x_train = x_train/255

x_test = x_test.reshape(-1, img_size, img_size, 1)
mean_test = np.mean(x_test, axis=0)
x_test = x_test-mean_test
x_test = x_test/255

print(x_train.shape)
print(x_test.shape)

# convert label to categorical
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Parameters settings
filters_numbers = [8, 16, 32]
filters_size = [[5,5],[4,4],[3,3]]
pool_size=(2, 2)
weight_decay = 5e-4
dropout = 0.5
lr = 0.001
momentum = 0.9
input_shape = x_train.shape[1:]

# Hyperparameters
epochs = 10 
batch_size = 32

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

history= model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.1)

#model.save('saved_models/fruits_classifier')

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
pyplot.savefig('fruits_classifier-training_acc.png')
pyplot.show()


pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('Model Loss')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.xticks(x)
pyplot.legend(['Train', 'Validation'], loc='upper right')
pyplot.grid(b=None, which='major', axis='both')

pyplot.savefig('fruits_classifier-training_loss.png')
pyplot.show()


# Test the model
print("[INFO] Evaluate the test data")

results = model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=2)
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

pyplot.savefig('fruits_classifier-confusion-matrix.png')
pyplot.show()
