import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

old_positive_angle = 0
old_positive_line = None
old_negative_angle = 0
old_negative_line = None


def process(frame):
    image = frame
    #image = set_binary(frame)
    image = detect_edges(image, 50, 300)
    image = gaussian_blur(image, 5, 5)
    image = set_roi(image)
    image = hough_lines(image, frame)
    return image


def set_binary(orig):
    # WHITE
    img = orig
    # Lower and upper bounds for colour values
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([145, 60, 255])

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(img, white_lower, white_upper)

    # YELLOW
    img = orig
    # Lower and upper bounds for colour values
    yellow_lower = np.array([80, 80, 200])
    yellow_upper = np.array([255, 255, 255])

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask2 = cv2.inRange(img, yellow_lower, yellow_upper)

    mask = cv2.bitwise_or(mask1, mask2)
    res = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    (thresh, image) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    return image


def gaussian_blur(img, x, y):
    gauss_img = cv2.GaussianBlur(img, (x, y), 0)
    return gauss_img


def detect_edges(img, lower, upper):
    can_img = cv2.Canny(img, lower, upper)
    return can_img


def set_roi(img):
    imshape = img.shape

    # points of the roi
    lower_left = [0, imshape[0] - imshape[0]/4]
    lower_right = [imshape[1], imshape[0] - imshape[0]/4]
    top_left = [imshape[1]/2 - imshape[1]/10, imshape[0]/2 - imshape[0]/10]
    top_right = [imshape[1]/2 + imshape[1]/10, imshape[0]/2 - imshape[0]/10]

    vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # mask with depending on the input image's colour channels
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def hough_lines(img, frame):
    global old_positive_angle
    global old_negative_angle
    global old_positive_line
    global old_negative_line

    lines = cv2.HoughLinesP(img, 1, math.pi / 180.0, 40, np.array([]), 100, 200)

    try:
        # Line lengths
        longest_positive_length = 0
        longest_negative_length = 0
        # Lines
        longest_positive_line = None
        longest_negative_line = None

        draw_positive_line = False
        draw_negative_line = False

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # get the lines angle
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if angle < -20 or angle > 20:
                # get the longest positive and negative line
                length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if angle > 0:
                    if length > longest_positive_length:
                        longest_positive_length = length
                        longest_positive_line = line
                elif angle < 0:
                    if length > longest_negative_length:
                        longest_negative_length = length
                        longest_negative_line = line

        if longest_positive_length != 0 and longest_negative_length != 0:
            # Positive Line
            xp1, yp1, xp2, yp2 = longest_positive_line[0]
            # if old_positive_angle == 0:
            #     old_positive_angle = np.arctan2(yp2 - yp1, xp2 - xp1) * 180 / np.pi
            #     old_positive_line = longest_positive_line
            # else:
            #     new_positive_angle = np.arctan2(yp2 - yp1, xp2 - xp1) * 180 / np.pi
            #     if not new_positive_angle > old_positive_angle + 2 or new_positive_angle < old_positive_angle - 2:
            #         xp1, yp1, xp2, yp2 = old_positive_line[0]
            # Negative Line
            xn1, yn1, xn2, yn2 = longest_negative_line[0]
            # if old_negative_angle == 0:
            #     old_negative_angle = np.arctan2(yn2 - yn1, xn2 - xn1) * 180 / np.pi
            #     old_negative_line = longest_negative_line
            # else:
            #     new_negative_angle = np.arctan2(yn2 - yn1, xn2 - xn1) * 180 / np.pi
            #     if not new_negative_angle < old_negative_angle - 2 or new_negative_angle > old_negative_angle + 2:
            #         xn1, yn1, xn2, yn2 = old_negative_line[0]

            # Draw lane lines
            cv2.line(frame, (xp1, yp1), (xp2, yp2), (0, 0, 255), 3)
            cv2.line(frame, (xn1, yn1), (xn2, yn2), (0, 0, 255), 3)

            # Draw center line
            topX = int((xp1 + xn2) / 2)
            botX = int((xp2 + xn1) / 2)
            cv2.line(frame, (topX, yp1), (botX, yp2), (0, 255, 0), 3)

    except Exception as e:
        print(e)

    return frame


