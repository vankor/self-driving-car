# Behavioral Cloning for Self Driving Car
The goal of this project is to build deep neural network to predict steering angle of car on track in udacity simulator.
Simulator was being used for collecting training data from 3 cameras. 

## Files and Usage
`model.py` : training codes.
`image_augmentation.py` : code related to image preprocessing
`driver_model`: trained model
`drive.py`: code that runs driving simulation
`video.mp4`: video on test track
How to use:
- open Udacity simulator and select track (autonomous)
- in cmd execute "python drive.py driver_model"

### Generate training data, training data specific
Data is collected from 3 cameras : left, right and center. Udacity dataset was being used because hands driving is not so smooth. 
There are 2 important points that should be understood about initial data:
1. Steering angle values from side cameras should be corrected (in +0.25 and -0.25 for left and right)
2. If value of steering angle > 0.2 it is treated like turning right, if angle < -0.15 is turning left. 
If value is close to zero it is treated like moving straight. 
3. In current project we only have to predict angle, so throttle, brake and speed are ignored.

### Data Augmentation
Following data augmentation techniques are applied in current project:

1. Random brightness: each image is converted to HSV colorspace and random brightness is obtained by adjusting V.
2. Random flipping: With 50% probability each image is flipped horizontally.
3. Random warping: each image is warped randomly (shifted horizontally) and steering angle is adjusted respectively to shift. In such way training set becomes more general and includes more possible turns and angles.
4. Each image is preprocessesd - cropped (only region of interest is being used for learning) and resized to 200 X 66 size for being competible with NVidia convnet architecture.
Below is example of augmented images:

### Model training and Validation
Initial data was splitted using `train_test_split` method of `sklearn` library. Proportion is 80/20.
Model was trained using fit_generator() methods. For learning process optimization generators were being used for training data and validation data. Training data generator is responsible for reading images, executing of random augmentation and preprocessing. Finally it provides training batches to network. Validation generator reads images, makes only preprocessing (like cropping and resize) and provides batches to network.
The data set was randomly shuffled. 

### Training
Nvidia convnet architecture was being applied in current project. But it was improved by adding extra dropout layers to avoid overfitting. So architecture is following:
    - 5X5 conv filter 24 filter depth, activation - 'relu' 
    - Dropout(0.5)
    - 5X5 conv filter 36 filter depth, activation - 'relu'
    - Dropout(0.5)
    - 5X5 conv filter 48 filter depth, activation - 'relu'
    - Dropout(0.5),
    - 3X3 conv filter 64 filter depth, activation - 'relu'
    - 3X3 conv filter 64 filter depth, activation - 'relu'
    - Flatten(),
    - Dense layer(100),
    - Dense layer (50),
    - Dense layer (10),
    - Dense layer (1)

Adam optimizer was being used with default learning rate. Optimal number of epoch is 3. More epochs give worse accuracy and car behavior on track is not accurately as well. Batch size = 64. 

### Testing
Trained model is saved in 'driver_model' file. Final model has loss about 0.015 for test set. Car behavior on track is quite smooth and car is able to drive even with high speed (was tested with speed = 20).

To run test: `python drive.py driver_model`


