# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 06:08:17 2019

@author: XIE
"""

import cnn
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.metrics import confusion_matrix



#
img_width, img_height = 80, 80
train_data_dir = 'train_data'
validation_data_dir = 'validation_data'
epochs = 12
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#
training_data_gen = ImageDataGenerator(rescale=1./255)
training_data_img = training_data_gen.flow_from_directory(
        train_data_dir, 
        batch_size=batch_size, 
        target_size=(img_width, img_height))

validation_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_img = validation_data_gen.flow_from_directory(
        validation_data_dir, 
        batch_size=batch_size, 
        target_size=(img_width, img_height))

#
model = cnn.create_model(input_shape, len(training_data_img.class_indices))

model.compile(optimizer=keras.optimizers.adam(), 
              loss=keras.losses.categorical_crossentropy, 
              metrics=['accuracy'])

model.summary()

model.fit_generator(
        training_data_img, 
        steps_per_epoch=training_data_img.samples / batch_size, 
        validation_data=validation_data_img, 
        validation_steps=validation_data_img.samples / batch_size,
        epochs=epochs, 
        verbose=1)

predictions = model.predict_generator(
        generator=validation_data_img, 
        steps=validation_data_img.samples / batch_size,
        verbose=1)

#
predict_classes = np.argmax(predictions, axis=1)
true_classes = validation_data_img.classes
class_labels = list(validation_data_img.class_indices.values())
matrix = confusion_matrix(true_classes, predict_classes, class_labels)
print(validation_data_img.class_indices)

#confusion matrix
print(matrix)
