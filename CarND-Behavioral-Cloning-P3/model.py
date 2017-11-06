import pickle
import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten, Dense, Activation, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout
from keras.models import Model, Sequential
import csv
import math
import cv2
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle, randint, uniform
from image_augmentation import plotImages, augmentImage, preprocessImage
import shutil
import os

samples_folder = 'data'
model_name = 'complex_generators_6'

def buildDistributionHist(angle_vals, title):
    plt.figure(figsize=(15, 5))
    plt.title(title)
    plt.xticks([i for i in np.arange(-1, 1, 0.1)]);
    plt.hist(angle_vals)
    plt.show()


def plotTrainProcess(h_object):
    plt.plot(h_object.history['loss'])
    plt.plot(h_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def showRandomImages(lines):
    randind1 = randint(0, len(lines))
    randind2 = randint(0, len(lines))
    randind3 = randint(0, len(lines))
    line1 = lines[randind1]
    line2 = lines[randind2]
    line3 = lines[randind3]
    img1, angle1 = cv2.imread('./' + samples_folder + '/IMG/' + line1[0].split('/')[-1]), float(line1[3])
    img2, angle2 = cv2.imread('./' + samples_folder + '/IMG/' + line2[0].split('/')[-1]), float(line2[3])
    img3, angle3 = cv2.imread('./' + samples_folder + '/IMG/' + line3[0].split('/')[-1]), float(line3[3])
    img_new1, angle_new1 = augmentImage(img1, angle1)
    img_new2, angle_new2 = augmentImage(img2, angle2)
    img_new3, angle_new3 = augmentImage(img3, angle3)
    plotImages([(img1, angle1), (img2, angle2), (img3, angle3), (img_new1, angle_new1), (img_new2, angle_new2),
                (img_new3, angle_new3)], cols=3)


#generator for training and validation sets
def generator(samples, batch_size=32 , isTrain = True):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                i = round(uniform(0, 2))
                current_image_path = batch_sample[i]
                filename = current_image_path.split('/')[-1]
                current_image = cv2.imread('./' + samples_folder + '/IMG/' + filename)
                angle = float(batch_sample[3])
                if isTrain:
                    if i == 1:
                        angle += 0.25
                    if i == 2:
                        angle -= 0.25
                    current_image, angle =  augmentImage(current_image, angle)
                else:
                    current_image = preprocessImage(current_image)
                images.append(current_image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)




lines = []
with open("./"+samples_folder+"/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    first_row = next(reader)
    for line in reader:
        lines.append(line)


showRandomImages(lines)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

batch_size = 64
epochs = 3

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size, isTrain=False)

#shape = (160, 320, 3)
shape = (66, 200, 3)
model = Sequential([
    Lambda(lambda x: (x / 255.0) - 0.5, input_shape=shape),
    Convolution2D(24,5,5, subsample = (2,2), activation = "relu"),
    Dropout(0.5),
    Convolution2D(36,5,5, subsample = (2,2), activation = "relu"),
    Dropout(0.5),
    Convolution2D(48,5,5, subsample = (2,2), activation = "relu"),
    Dropout(0.5),
    Convolution2D(64,3,3, activation = "relu"),
    Convolution2D(64,3,3, activation = "relu"),
    Flatten(),
    Dense(100),
    Dense(50),
    Dense(10),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data =
                                     validation_generator,
                                     nb_val_samples = len(validation_samples),
                                     nb_epoch=epochs, verbose=1)


plotTrainProcess(history_object)
model.save(model_name)
