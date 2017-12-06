# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/CarImages.png
[image11]: ./examples/NonCarImages.png
[image2]: ./examples/HOG_examples.png

[image3]: ./examples/windows_example.jpg
[image4]: ./output_images/outfile_windowed_1.jpg
[image41]: ./output_images/outfile_windowed_2.jpg
[image42]: ./output_images/outfile_windowed_3.jpg
[image43]: ./output_images/outfile_windowed_5.jpg

[image5]: ./output_images/outfile_heated_1.jpg
[image51]: ./output_images/outfile_heated_2.jpg
[image52]: ./output_images/outfile_heated_3.jpg
[image53]: ./output_images/outfile_heated_5.jpg

[image7]: ./output_images/outfile_labeled_1.jpg
[image71]: ./output_images/outfile_labeled_2.jpg
[image72]: ./output_images/outfile_labeled_3.jpg
[image73]: ./output_images/outfile_labeled_5.jpg


[video1]: ./project_video_final_processed.mp4
[video2]: ./project_video_final_processed_yuv.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Reading training examples and experiment with color spaces

Firstly training examples are loaded for cars and non cars images. Pls see section 1 in pipeline.py file.

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image11]

On the next step different `skimage.hog()` parameters were explored (like color space, orient, pixels per cell, cells per block, different channels).

I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Below is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Choosing final HOG parameter values

I tried various combinations of HOG parameters . 
And the best chooses for me were:

- YCrCb color space, 9 orient, 8 pixels per cell, 2 cells per block, using all color channels for gradient 
- YUV color space, 11 orient, 16 pixels per cell, 2 cells per block, using all color channels for gradient
- HLS color space, 9 orient, 8 pixels per cell, 2 cells per block, using all color channels for gradient

YCrCb color space provides quite good performance provides, so it has been picked.

#### 3. Using SVM classifier

I trained a linear SVM using extracted HOG features as training set. Spatial features 
and color histogram features were being tried but they dont give significant impact to performance. 
That's why only HOG features were being used for SVM model input.
Pls see section 3 in pipeline.py file. 
Accuracy of final trained model is about 98.5%. Sounds good

### Sliding Window Search

#### 1. Window scales, overlapping and region of interest

Region of interest where vehicles is between 400 and 660 on y axis. Sliding windows was using only for this area. 
Area closer to bottom of ROI was slided with 3.5 scaled windows, area closer to top of ROI was slided with scale 1.0. 
Middle area of ROI was scanned with 1.5-2.0 scaled windows. Windows were configured in following way (ystart, ystop, scale):

`window_sizes = [(400, 464, 1.0),
                 (420, 580, 1.5),
                 (400, 660, 1.5),
                 (400, 660, 2.0),
                 (500, 660, 3),
                 (464, 660, 3.5)]` 
                 
Window overlapping was being used 0.5 for x and 0.5 for y axis.
Pls see sliding window implementation in section 4 in pipeline.py file and 'find_cars' function in functions.py file.
Below is demonstrated how are windows looks like:

![alt text][image3]

#### 2. Test processed images example 

Ultimately frames were scanned with sliding windows using YCrCb 3-channel HOG features. 
Results are showed below:

![alt text][image4]

![alt text][image41]

![alt text][image42]

![alt text][image43]

---

### Video Implementation

#### 1. Video results
Here's a [link to my video result](./project_video_final_processed.mp4)
Also you can see video with features extracted for YUV color space [link to my video result (yuv colorspace)](./project_video_processed_yuv.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Positions of positive detections were marked with rectangles in each frame of the video.  
From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  
`scipy.ndimage.measurements.label()` was being used to identify individual blobs in the heatmap.  
There is assumed that each blob corresponds to a vehicle.  
To obtain shape of rectangle bounding boxes were constructed for the area of each blob detected.  


### Here are corresponding heatmaps for frames demonstrated above:

![alt text][image5]
![alt text][image51]
![alt text][image52]
![alt text][image53]


### Here is the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]
![alt text][image71]
![alt text][image72]
![alt text][image73]

---

### Discussion

#### 1. Problems and improvements

Current implementation of model has some disadvantages like:
 - there may be some false positive examples when there are noise or shadows
 - there are vehicles from opposite lane of road detected
 - there could be troubles when many vehicles will be on the road
 
Possible improvements could be:
 - more restrictive ROI (maybe x axis also can be restricted)
 - more complex learning algorithm (maybe using of neural network can be useful)
 - further experiments with features extraction (introduce new features, combine features, combine colorspaces)
