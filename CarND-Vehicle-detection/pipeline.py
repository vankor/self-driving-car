import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import scipy.misc
from functions import *
from scipy.ndimage.measurements import label
from sklearn.externals import joblib
from moviepy.editor import VideoFileClip


# SECTION 1 - Read training data for cars and notcars

images_nv = glob.glob('training_set/non-vehicles/**/*.png', recursive=True)
images_v = glob.glob('training_set/vehicles/**/*.png', recursive=True)

cars = []
notcars = []
for image in images_nv:
    notcars.append(image)

for image in images_v:
    cars.append(image)


# SECTION 2 - Parameter selection and extracting features

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9 # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off

car_features = extract_features(cars, cspace=color_space, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

notcar_features = extract_features(notcars, cspace=color_space, orient=orient,
                                   pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
#scaler = StandardScaler().fit(X)
#scaled_X = scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# SECTION 3 - Prepare model, train model, save model

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
      'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

joblib.dump(svc, './saved_models/classifier_YCrCb.pkl')


# SECTION 4 - Configure and implement sliding windows, implement drawing rectangles, smoothing for videos
window_sizes = [(400, 464, 1.0),
                (420, 580, 1.5),
                (400, 660, 1.5),
                (400, 660, 2.0),
                (500, 660, 3),
                (464, 660, 3.5)]

rectangles_sequence = []

def smooth_rects(rects):
    global rectangles_sequence
    rectangles_sequence.append(rects)
    if len(rectangles_sequence) > 12:
        # throw out oldest rectangle set(s)
        rectangles_sequence = rectangles_sequence[len(rectangles_sequence) - 12:]


def detect_vehicles(image):
    draw_image = np.copy(image)

    hot_windows = slide_multiple_windows(image, window_sizes, svc, color_space,
                                         orient, pix_per_cell, cell_per_block,
                                         spatial_size, hist_bins, None)


    if len(hot_windows) > 0:
        smooth_rects(hot_windows)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    for rect_set in rectangles_sequence:
        heat = add_heat(heat, rect_set)
    heat = apply_threshold(heat, 1 + len(rectangles_sequence) // 2)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    result_img, rects = draw_labeled_bboxes(np.copy(draw_image), labels)
    return result_img


def run_for_test_images():
    i = 1
    example_images = glob.glob('test_images/*.jpg')
    for file in example_images:
        image = mpimg.imread(file)
        draw_image = np.copy(image)
        hot_windows = slide_multiple_windows(image, window_sizes, svc, color_space,
                                             orient, pix_per_cell, cell_per_block,
                                             spatial_size, hist_bins, None)

        image = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

        scipy.misc.imsave('output_images/outfile_windowed_' + str(i) + '.jpg', image)

        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        heat = add_heat(heat, hot_windows)

        scipy.misc.imsave('output_images/outfile_heated_' + str(i) + '.jpg', heat)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img, rects = draw_labeled_bboxes(np.copy(draw_image), labels)

        scipy.misc.imsave('output_images/outfile_labeled_' + str(i) + '.jpg', draw_img)
        i += 1

def run_for_video():
    test_out_file2 = 'project_video_final_processed.mp4'
    clip_test2 = VideoFileClip('project_video.mp4')
    clip_test_out2 = clip_test2.fl_image(detect_vehicles)
    clip_test_out2.write_videofile(test_out_file2, audio=False)



run_for_video()