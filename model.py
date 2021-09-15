
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import csv
from math import ceil
from random import shuffle

count=0
samples = []
with open('./centre_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        count+=1
        samples.append(line)
with open('./side_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        count+=1
        samples.append(line)
with open('./reverse_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        count+=1
        samples.append(line)
with open('./new_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        count+=1
        samples.append(line)
with open('./lastest_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        count+=1
        samples.append(line)
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        count+=1
        line[0]='/home/user/Documents/sd_car_nanodegree/CarND-Behavioral-Cloning-P3/data/'+line[0]
        samples.append(line)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #yield sklearn.utils.shuffle(X_train, y_train)
            yield (X_train, y_train)

# Set our batch size
batch_size=128

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320  # Trimmed image format

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda





import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np

np.random.seed(1000)





AlexNet = Sequential()
AlexNet.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
AlexNet.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#2nd Convolutional Layer
AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#3rd Convolutional Layer
AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

#4th Convolutional Layer
AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

#Passing it to a Fully Connected layer
AlexNet.add(Flatten())
# 1st Fully Connected Layer
AlexNet.add(Dense(64))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))


#Output Layer
AlexNet.add(Dense(1))

#Model Summary
AlexNet.summary()



AlexNet.compile(loss='mse', optimizer='adam')



AlexNet.fit_generator(train_generator,
            steps_per_epoch=ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=ceil(len(validation_samples)/batch_size),
            epochs=10, verbose=1)



AlexNet.save('model.h5')


