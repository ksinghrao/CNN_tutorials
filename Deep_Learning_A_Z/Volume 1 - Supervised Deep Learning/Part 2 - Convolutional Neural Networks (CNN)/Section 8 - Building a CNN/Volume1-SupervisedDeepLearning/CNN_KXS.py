#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:58:22 2017

@author: kamalsinghrao
"""

'''
CNN steps
You use CNNs to add layers to an ANN to preserve spatial strucutres of images for classification. 

Step 1: Convolution 
      - Feature detection using convolution
      - Use ReLu to activate certain feature maps
Step 2: Max pooling
      - Create a pooled feature map from the feature map. Pooled feature maps are smaller than feature maps. 
      - This introduces spatial and random feature invariance to the feature maps.
Step 3: Flattening
      - Flatten the pooled feature maps into a single vector so they are inputs to a artifical neural network
Step 4: Full connection
      - Feed each feature map to an ANN. 
      - During this step the weights in the ANN will be adjusted based on what it has learnt. 

Softmax - used to normalize the probabilities of the final neural network output to 1. 

Cross Entropy is a method of calculating the error of a neural network.

If you flattened the input image and fed that into a ANN then you lose spatial information between the pixels. This is why you need the convolution and max pooling steps.
'''

'''
Seperate your images into a training set of images and a test set of images. Create a subfolder of a dog and cat. By doing this the dataset is split into the test set and training set. 

Feature scaling does need to be performed. 

A downside of overfitting is that you cannot generalize the results of your model (e.g. too few training sets)
'''
# Part 1 - Bulding the CNN
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#these 2 lines are for timer
from timeit import default_timer as timer
start = timer()

# Initialising the CNN
classifier = Sequential()

# Step 1 and 2 - Convolution layer and max pooling
classifier.add(Conv2D(32,(3,3),activation='relu',input_shape = (64,64,3)))
classifier.add(MaxPooling2D(pool_size = (2,2)))

'''
On a GPU you can add 64 filters for each pixel
'''

# Add a second convolutional layer  
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection layer
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])

# Part 2. Fitting the CNN to the images
'''
For smaller datasets you can use data/image augmentations (subject to random transformations) to use smaller datasets with less overfitting. 
'''
# Image augmentation options
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2, # random zooms
        horizontal_flip=True) # horizontal image flips

test_datagen = ImageDataGenerator(rescale=1./255)

# Apply image augmentation, train and test image sets
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000, # number of images in training set
        epochs=5,
        validation_data=test_set,
        validation_steps=2000) # number of images in test set

# Save the classifier
import json
saved_model = classifier.to_json()
with open('classifier.json', 'w') as outfile:
        json.dump(saved_model, outfile)
        
 
## Load model
#from keras.models import model_from_json
#
#with open('classifier.json', 'r') as architecture_file:    
#    model_architecture = json.load(architecture_file)
# 
# 
#classifier = model_from_json(model_architecture)

# Make a single prediction
import numpy as np
from keras.preprocessing import image
test_image1 = image.load_img('Single test images/cat.jpg',target_size = (64,64))
test_image1 = image.img_to_array(test_image1)
# Add an extra dimension to denote the batch number
test_image1 = np.expand_dims(test_image1, axis =0)
result = classifier.predict(test_image1)
training_set.class_indices
if result[0][0] ==1:
    prediction = 'dog'
else:
    prediction = 'cat'    
    
print(prediction)