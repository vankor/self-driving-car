# **Finding Lane Lines on the Road** 

## 1. Algorithm pipeline
### Basic pipeline consists of 7 steps:
 - grayscale the source image
 - filter out 'noise' colors from grayscaled image (only range from 180 to 255 is representative)
 - apply gausain blur with kernel size = 5 
 - apply canny edge detection with following params: low_threshold = 80, high_threshold = 210
 - define vertices and find representative region on images where most likely will be lanes
 - find lines on image using Hough space transform algorithm. Parameters are: rhu = 2, theta = 1, threshold = 22, min_line_length = 6, min_gap_between_lines = 5
 - use upgraded draw_line() function in order to draw a single line on the left and right lanes
 - put detected lines on the initial image

### Custom draw_lines() function does the following steps:
- calculates positive/negative slopes for left/right lanes 
- filters out anomaly sloped lines (with slope < 0.5)
- calculates average slopes (m) separate for negative and positive lines based on top n longest lines
- calculates b coefficient value (using formula: y = mx + b -> b = y - mx)
- builds two straight lane lines based on above calculated info
- draws constructed lines

## 2. Potential shortcomings
One of potential shortcomings would happen when there will be many 'noises' and obstacles on image or video stream. 
Kinds of possible noises/obstacles are: shadows, glares, another cars intersecting lane lines, pedestrians, etc
Some redundant incorrectly detected lines can be token into account calculating average lane lines and increase algorithm error.

Yet another shortcoming is that road markup could be blurred, unclear or drawn with unusual color. Then algorithm could face troubles detecting it. 

Another shortcoming could be lane detection on abrupt turns when lane lines can be out of predefined vertices on image and as well have unusual slopes. This trouble could raise on intersections or during active car maneuvering.

## 3. Possible improvements to pipeline

A possible improvement would be to add more advanced image preprocessing: removing noises, glares and shadows from source image, obstacles detection and removing  from source image. This will reduce redundant 'noise' lines count and improve detection accuracy.

One more possible improvement could be dynamic vertices, margin allowed slopes and other params adjustment depending on car maneuvering. For example when car turns. This requires to know info about earlier state images to detect maneuvering type and adjust param values.

Also one more improvement based on having previous states images will be possibility to filter out outliers. Algorithm will be able to calculate probability of each lane slope value based on previous images and use most likely one.

Another potential improvement could be to teach algorithm for some kind of approximation of where lane lines may be and which slopes they may have. This algorithm can be based on machine learning techniques like neural networks, regressions, etc.
