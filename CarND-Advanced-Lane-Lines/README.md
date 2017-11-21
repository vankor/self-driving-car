## Advanced Lane Lines Finding

**Advanced Lane Finding Project**

All the code of this project is in 'lane_lines_detector.ipynb' notebook.

[//]: # (Image References)

[image1]: ./images/chess_corners.png "Multiple calib"
[image20]: ./images/chess_calib.png "Road Transformed"
[image2]: ./images/calibration_example.png "Road Transformed"
[image3]: ./images/multiple_threshold_examples.png "Binary Example"
[image4]: ./images/before_after_warp.png "Warp Example"
[image5]: ./images/lines_finding_step1.png "Preprocess"
[image6]: ./images/lines_finding_ex2.png "Fitted lines"
[image7]: ./images/lines_finding_ex2.png "Fitted lines"
[image8]: ./images/challenge01.jpg_processed.jpg "Processed1"
[image9]: ./images/test1.jpg_processed.jpg "Processed2"
[image10]: ./images/test6.jpg_processed.jpg "Processed3"
[video1]: ./project_video_output_final.mp4 "Video"

### Camera calibration

The code for this step is contained in the third code cell of the IPython 'lane_lines_detector.ipynb'  notebook.
Firstly object points were prepared. `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. Below you can see corners detection results:  

![alt text][image1]

### Pipeline (single images)

#### 1. Image distortion correction.

`objpoints` and `imgpoints` are being used to compute the camera calibration matrix and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function. For validating distortion correction process chess board image was being used. Below you can see how distortion correction looks like:
![alt text][image20]![alt text][image2]

#### 2. Color spaces transformation and using image thresholdes .

Function 'combine_thresholds' is responsible for image colors transformation. There were grayscale threshold, HSL and LAB colorspaces being used. In HSL colorspace threshold is applied to L and S channels and in LAB colorspace threshold was being applied for B channel.

I have tried to use color gradient (Sobel function) but it performs not so well as manipulation with colorspaces and thresholds, especially LAB and B channel. That has helped a lot to reduce noise on the road and shadows.
Below you can see example of using colorspace transformations and channel thresholds for multiple images from all project videos:

![alt text][image3]

#### 3. Perspective transform 

Function 'image_preprocessing_pipeline' does perspective transformation using transformation matrix returned from 'build_transformation_matricies' function. Source and destination points for warping image were hardcoded following way:

```python
src = np.array([[585. /1280.*img_size[1], 455./720.*img_size[0]],
                        [705. /1280.*img_size[1], 455./720.*img_size[0]],
                        [1130./1280.*img_size[1], 720./720.*img_size[0]],
                        [190. /1280.*img_size[1], 720./720.*img_size[0]]],
                        np.float32)

dst = np.array([[300. /1280.*img_size[1], 100./720.*img_size[0]],
                        [1000./1280.*img_size[1], 100./720.*img_size[0]],
                        [1000./1280.*img_size[1], 720./720.*img_size[0]],
                        [300. /1280.*img_size[1], 720./720.*img_size[0]]],
                        np.float32)
```
Perspective transform is verified by drawing src points on image. Below you can see how perspective warping works:

![alt text][image4]

#### 4. Lane-line pixels identification and building polynomial fits for two lane lines

From scratch if there are no pixels available on previous steps histogram sliding window technique was being applied to identify needed pixels for each line. See function 'build_lines_from_scratch' in 'lane_lines_detector.ipynb' notebook. 

If there are already some left and right line pixels identified on previous steps they can be used to get new lane lines pixels. See function 'build_lines_from_prev_points' that build new lanes pixels based on where are previously defined lanes pixels using margin.

![alt text][image5]
![alt text][image6]

#### 5. Calculation of lane curvature radius and identification of the vehicle position with respect to center
Radius of lane curvature was calculated using given formula.

#### 6. Another techniques
- There was being used smoothing using moving average for lane fits. This helps to make lane detection more accurate.
- Also, sanity check was applied to filter out fits with anomal coefficients. For this purposes was being used difference in fit coefficients between last and new fits.

#### 7. Drawing lane area on the image 
Function 'draw_lines()' is responsible for drawing lane area on the original image. Below you can see drawn lane areas on images from different videos:

![alt text][image8]
![alt text][image9]
![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There were following problems faced during implementation:
1. Sometimes there was not enough points to fit one of the lines correctly
2. Noise and shadows on images
3. Another vehicle on road 
