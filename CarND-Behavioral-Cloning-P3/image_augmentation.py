import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from random import uniform
import math

def plotImages(images_to_labels, predictions = False, cols=3, sign_names=None):
    rows = len(images_to_labels)//cols
    plt.figure(figsize=(cols*3,rows*2.5))
    i = 0
    for (image_file, label) in images_to_labels:
        print(label)
        plt.subplot(rows, cols, i+1)
        plt.imshow(image_file.squeeze(), cmap="gray")
        plt.xticks([])
        plt.yticks([])
        if label is not None and not predictions:
            plt.text(0, image_file.shape[1] + 3, '{}'.format(label),
                     fontsize=12, color='k',backgroundcolor='w')
        if sign_names is not None and predictions:
            plt.text(0, 0, 'Predicted: {}({})'.format(sign_names[label], label),
                     fontsize=8, color='k', backgroundcolor='y')
        i = i + 1
    plt.show()


def flip(img, angle):
    img = cv2.flip(img, 1)
    angle = -angle
    return img, angle

def crop(image, y1, y2, x1, x2):
    return image[y1:y2, x1:x2]

def randFlip(img, angle):
    if uniform(0, 1) > 0.5:
        img, angle = flip(img, angle)
    return img, angle

def augment_brightness(image):
    new_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    new_image = np.array(new_image, dtype = np.float64)
    bright_delta = np.random.uniform() + 0.5
    new_image[:,:,2] = new_image[:,:,2] * bright_delta
    new_image[:,:,2][new_image[:,:,2]>255]  = 255
    new_image = np.array(new_image, dtype = np.uint8)
    new_image = cv2.cvtColor(new_image,cv2.COLOR_HSV2RGB)
    return new_image

def randWarp(image, angle, transition_value):
    rows, cols, _ = image.shape
    warp_x = transition_value * np.random.uniform() - transition_value / 2
    M = np.float32([[1,0,warp_x],[0,1, 0]])
    warped_image = cv2.warpAffine(image, M, (cols,rows))
    image = crop(warped_image, 20, 140, 0 + transition_value, cols - transition_value)
    new_angle = (0.4 * (warp_x / transition_value)) + angle
    return image,new_angle

def augmentImage(image, y_steer):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image, y_steer  = randWarp(image, y_steer, 100)
    if abs(y_steer) < 0.2:
        y_steer = 0
    image = augment_brightness(image)
    image, y_steer = randFlip(image, y_steer)
    image = preprocessImage(image)
    return image, y_steer

def preprocessImage(image):
    shape = image.shape
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    return cv2.resize(image,(200, 66), interpolation=cv2.INTER_AREA)




