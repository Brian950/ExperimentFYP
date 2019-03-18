import cv2
import glob
import numpy as np
from NanMa_et_al.lane_detection import LaneDetection
from NanMa_et_al.line import Line
import time
from matplotlib import pyplot as plt

SOBEL = True
img_size = (854, 360)

lane_params = {'saturation': 100, 'light_yellow': 60, 'light_white': 195,
               'gradient': (0.7, 1.5), 'x_thresh': 20, 'magnitude': 40}
lane_finder = LaneDetection(lane_params)

left_lane = False
right_lane = False

left_line = Line()
right_line = Line()

sharp_kernel = np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]])


def process(frame, mtx=None, dist=None, selection=None):

    image = crop(frame, 0, 360, 0, 854)

    # mtx set to None temporarily due to issue with distort leaving black pixels.
    mtx = None
    if mtx is not None and dist is not None:
        view = undistort(image, mtx, dist)
    else:
        view = image

    src, dst = get_perspective_matrix(selection)

    transform = perspective_transform(view, src, dst, img_size)

    # Classification of light intensity
    gray = cv2.cvtColor(transform, cv2.COLOR_RGB2GRAY)
    gray_mean_val = np.mean(gray)

    #Pixel gray value
    if gray_mean_val > 150:
        # High light intensity = gray equalisation
        eq_image = gray_equalisation(gray)
    else:
        # Low light intensity = gray stretching
        eq_image = gray

    sharp = sharpen(eq_image)
    sharp = sharpen(eq_image)
    gb_sharp = gaussian_blur(sharp, 7, 7)

    #sobel_image = sobel(gb_sharp, 1)
    #sobel_image = gaussian_blur(sobel_image, 31, 31)

    canny_image = canny(eq_image, 200, 200)

    #binary_image = set_binary(transform)
    gray_binary_image = set_gray_binary(gb_sharp)

    #combo_image = cv2.bitwise_or(canny_image, binary_image)

    color_binary_image = lane_finder.apply(transform)

    combo_image = cv2.bitwise_or(gray_binary_image, color_binary_image)

    output_img = find_lanes(combo_image)
    res, clear_distort = draw_lane(view, transform, output_img, src, dst)

    res_combo = cv2.bitwise_or(view, res)

    result = assemble_img(clear_distort, output_img, res_combo, combo_image)

    return result


def get_perspective_matrix(selection):

    if selection == "highway.mp4":
        src = np.float32([[0, 0], [854, 0], [0, 360], [854, 360]])
        dst = np.float32([[320, 240], [550, 240], [0, 360], [854, 360]])
        return src, dst
    elif selection == "highway_sunlight.mp4":
        src = np.float32([[0, 0], [854, 0], [0, 360], [854, 360]])
        dst = np.float32([[260, 210], [500, 210], [0, 360], [854, 360]])
        return src, dst
    else:
        src = np.float32([[0, 0], [854, 0], [0, 360], [854, 360]])
        dst = np.float32([[260, 210], [500, 210], [0, 360], [854, 360]])
        return src, dst


def set_gray_binary(image):
    ret, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    return thresh


def crop(image, ht, hb, wt, wb):
    image = image[ht:hb, wt:wb]
    return image


def gaussian_blur(img, x, y):
    gauss_img = cv2.GaussianBlur(img, (x, y), 0)
    return gauss_img


# Calibration method modified from https://opencv-python-tutroals.readthedocs.io
def calibrate_camera(directory, nx, ny, img_size):
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob.glob(directory+'*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    return mtx, dist


def undistort(frame, mtx, dist):
    frame = cv2.undistort(frame, mtx, dist, None, mtx)
    return frame


def gray_equalisation(image):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    image = clahe.apply(image)
    return image


def perspective_transform(image, src, dst, img_size):
    matrix = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, matrix, img_size)
    return warped


def sobel(image, kernel):
    image = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel)
    return image


def canny(image, lower, upper):
    image = cv2.Canny(image, lower, upper)
    return image


def sharpen(image):
    image = cv2.filter2D(image, -1, sharp_kernel)
    image = cv2.bilateralFilter(image, 5, 50, 50)
    return image


def window_search(binary_warped):
    # Take a histogram of the bottom half of the image
    bottom_half_y = binary_warped.shape[0]/2
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 8
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 30
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Generate black image and colour lane lines
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 1, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 1, 0]

        # Draw polyline on image
        right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
        left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
        cv2.polylines(out_img, [right], False, (1, 0, 1), thickness=5)
        cv2.polylines(out_img, [left], False, (1, 0, 1), thickness=5)
    except:
        pass

    return left_lane_inds, right_lane_inds, out_img


def find_lanes(img):
    # Perform a full window search if no prior successful detections.
    # Window Search
    left_lane_inds, right_lane_inds, out_img = window_search(img)
    # Update the lane detections
    validate_lane_update(img, left_lane_inds, right_lane_inds)
    return out_img


def validate_lane_update(img, left_lane_inds, right_lane_inds):
    # Checks if detected lanes are good enough before updating
    img_size = (img.shape[1], img.shape[0])

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Extract left and right line pixel positions
    left_line_allx = nonzerox[left_lane_inds]
    left_line_ally = nonzeroy[left_lane_inds]
    right_line_allx = nonzerox[right_lane_inds]
    right_line_ally = nonzeroy[right_lane_inds]

    # Discard lane detections that have very little points,
    # as they tend to have unstable results in most cases
    if len(left_line_allx) <= 500 or len(right_line_allx) <= 500:
        left_line.detected = False
        right_line.detected = False
        return

    left_x_mean = np.mean(left_line_allx, axis=0)
    right_x_mean = np.mean(right_line_allx, axis=0)
    lane_width = np.subtract(right_x_mean, left_x_mean)

    # Discard the detections if lanes are not in their respective half of their screens
    if left_x_mean > 450 or right_x_mean < 350:
        left_line.detected = False
        right_line.detected = False
        return

    # Discard the detections if the lane width is too large or too small
    if lane_width < 200 or lane_width > 550:
        left_line.detected = False
        right_line.detected = False
        return

        # If this is the first detection or
    # the detection is within the margin of the averaged n last lines
    if left_line.bestx is None or np.abs(np.subtract(left_line.bestx, np.mean(left_line_allx, axis=0))) < 100:
        left_line.update_lane(left_line_ally, left_line_allx)
        left_line.detected = True
    else:
        left_line.detected = False
    if right_line.bestx is None or np.abs(np.subtract(right_line.bestx, np.mean(right_line_allx, axis=0))) < 100:
        right_line.update_lane(right_line_ally, right_line_allx)
        right_line.detected = True
    else:
        right_line.detected = False


def draw_lane(view, undist, img, src, dst):

    matrix = cv2.getPerspectiveTransform(src, dst)
    # Generate x and y values for plotting
    ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.stack((warp_zero, warp_zero, warp_zero), axis=-1)

    left_fit = left_line.best_fit
    right_fit = right_line.best_fit

    if left_fit is not None and right_fit is not None:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        clear_distort = undist.copy()

        # Draw the lane
        cv2.fillPoly(undist, np.int_([pts]), (64, 224, 208))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(undist, matrix, (img.shape[1], img.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(view, 1, newwarp, 0.6, 0)
        return newwarp, clear_distort
    return view, view


def assemble_img(warped, polynomial_img, lane_img, combo_image):
    # Define output image
    # Main image
    img_out = np.zeros((740, 1290, 3), dtype=np.uint8)
    img_out[0:360, 0:854, :] = lane_img

    combo_image = cv2.cvtColor(combo_image, cv2.COLOR_GRAY2RGB)
    # Text formatting
    fontScale = 1
    thickness = 1
    fontFace = cv2.FONT_HERSHEY_PLAIN

    # Perspective transform image
    img_out[0:240, 865:1285, :] = cv2.resize(warped, (420, 240))
    boxsize, _ = cv2.getTextSize("Transformed", fontFace, fontScale, thickness)
    cv2.putText(img_out, "Transformed", (int(1090 - boxsize[0] / 2), 40), fontFace, fontScale, (255, 255, 255),
                thickness, lineType=cv2.LINE_AA)

    #
    img_out[250:490, 865:1285, :] = cv2.resize(combo_image, (420, 240))
    boxsize, _ = cv2.getTextSize("Thresholding", fontFace, fontScale, thickness)
    cv2.putText(img_out, "Thresholding", (int(1090 - boxsize[0] / 2), 280), fontFace, fontScale, (255, 255, 255),
                thickness, lineType=cv2.LINE_AA)

    # Polynomial lines
    img_out[500: 740, 865:1285, :] = cv2.resize(polynomial_img * 255, (420, 240))
    boxsize, _ = cv2.getTextSize("Detected Lane", fontFace, fontScale, thickness)
    cv2.putText(img_out, "Detected Lane", (int(1090 - boxsize[0] / 2), 520), fontFace, fontScale, (255, 255, 255),
                thickness, lineType=cv2.LINE_AA)

    return img_out