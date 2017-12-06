import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features
def get_bin_spatial_features(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def get_color_hist_features(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes

    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy



def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    rects = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        rects.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image and final rectangles
    return img, rects


def convert_color(img, color_space='RGB'):
    if color_space != 'RGB':
        if color_space == 'HSV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: return np.copy(img)


# Define a single function that can extract features using hog sub-sampling and make predictions

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(image, ystart, ystop, scale, svc, hog_color_space, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, scaler):

    img = image.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]

    if hog_color_space == 'RGB':
        ctrans_tosearch = np.copy(image)
    else:
        ctrans_tosearch = convert_color(img_tosearch, hog_color_space)

    # rescale image if other than 1.0 scale
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)+1
    nyblocks = (ch1.shape[0] // pix_per_cell)+1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    rectangles = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Get color features
            features = []
            spatial_features = None
            hist_features = None
            if spatial_size is not None:
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                spatial_features = get_bin_spatial_features(subimg, size=spatial_size)
                features.append(spatial_features)
            if hist_bins is not None:
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                hist_features = get_color_hist_features(subimg, nbins=hist_bins)
                features.append(hist_features)
            if hog_features is not None:
                features.append(hog_features)

            # Scale features and make a prediction
            if scaler is not None and spatial_features is not None and hist_features is not None and hog_features is not None:
                test_features = scaler.transform(np.hstack(tuple(features)).reshape(1, -1))
            else:
                test_features = hog_features.reshape(1, -1)

            test_prediction = svc.predict(test_features.reshape(1, -1))

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                rectangles.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    return rectangles



def slide_multiple_windows(image, window_sizes, svc, color_space, orient, pix_per_cell,
                           cell_per_block, spatial_size, hist_bins, scaler):
    hot_windows = []

    for window_params in window_sizes:
        current_rectangles = find_cars(image, window_params[0], window_params[1], window_params[2], svc, color_space, orient, pix_per_cell, cell_per_block,
                                       spatial_size, hist_bins, scaler)
        if (len(current_rectangles) > 0):
            [hot_windows.append(rect) for rect in current_rectangles]

    return hot_windows



def draw_hog_examples(car_images, noncar_images):
    car_img = mpimg.imread(car_images[5])
    car_img_color = cv2.cvtColor(car_img, cv2.COLOR_RGB2YCrCb)
    _, car_dst = get_hog_features(car_img_color[:,:,2], 9, 8, 8, vis=True, feature_vec=True)

    noncar_img = mpimg.imread(noncar_images[5])
    noncar_img_color = cv2.cvtColor(car_img, cv2.COLOR_RGB2YCrCb)
    _, noncar_dst = get_hog_features(noncar_img_color[:,:,2], 9, 8, 8, vis=True, feature_vec=True)

    car_img2 = mpimg.imread(car_images[10])
    car_img_color2 = cv2.cvtColor(car_img2, cv2.COLOR_RGB2YCrCb)
    _, car_dst2 = get_hog_features(car_img_color2[:,:,2], 9, 8, 8, vis=True, feature_vec=True)

    noncar_img2 = mpimg.imread(noncar_images[10])
    noncar_img_color2 = cv2.cvtColor(noncar_img2, cv2.COLOR_RGB2YCrCb)
    _, noncar_dst2 = get_hog_features(noncar_img_color2[:,:,2], 9, 8, 8, vis=True, feature_vec=True)

    # Visualize
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, figsize=(7,7))
    f.subplots_adjust(hspace = .4, wspace=.2)
    ax1.imshow(car_img)
    ax1.set_title('Car image', fontsize=16)
    ax2.imshow(car_img_color)
    ax2.set_title('Car image in YCrCb', fontsize=16)
    ax3.imshow(car_dst, cmap='gray')
    ax3.set_title('Car after applying HOG', fontsize=16)

    ax4.imshow(car_img2)
    ax4.set_title('Car image', fontsize=16)
    ax5.imshow(car_img_color2)
    ax5.set_title('Car image in YCrCb', fontsize=16)
    ax6.imshow(car_dst2, cmap='gray')
    ax6.set_title('Car after applying HOG', fontsize=16)

    ax7.imshow(noncar_img)
    ax7.set_title('NonCar Image', fontsize=16)
    ax8.imshow(noncar_img_color)
    ax8.set_title('NonCar in YCrCb', fontsize=16)
    ax9.imshow(noncar_dst, cmap='gray')
    ax9.set_title('NonCar after applying HOG', fontsize=16)

    ax10.imshow(noncar_img2)
    ax10.set_title('NonCar Image', fontsize=16)
    ax11.imshow(noncar_img_color2)
    ax11.set_title('NonCar in YCrCb', fontsize=16)
    ax12.imshow(noncar_dst2, cmap='gray')
    ax12.set_title('NonCar after applying HOG', fontsize=16)

    plt.show()


def draw_train_examples(car_images, noncar_images):
    car_img = mpimg.imread(car_images[5])
    noncar_img = mpimg.imread(noncar_images[5])

    car_img2 = mpimg.imread(car_images[10])
    noncar_img2 = mpimg.imread(noncar_images[10])

    car_img3 = mpimg.imread(car_images[15])
    noncar_img3 = mpimg.imread(noncar_images[15])

    car_img4 = mpimg.imread(car_images[15])
    noncar_img4 = mpimg.imread(noncar_images[15])

    # Visualize
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7,7))
    f.subplots_adjust(hspace = .4, wspace=.2)
    f.suptitle('Car images', fontsize=16)
    ax1.imshow(car_img)
    ax2.imshow(car_img2)
    ax3.imshow(car_img3)
    ax4.imshow(car_img4)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7,7))

    f.suptitle('NonCar images', fontsize=16)
    ax1.imshow(noncar_img)
    ax2.imshow(noncar_img2)
    ax3.imshow(noncar_img3)
    ax4.imshow(noncar_img4)

    plt.show()
