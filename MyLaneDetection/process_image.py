import cv2
import numpy as np
import math
from Common_Tools.draw_lane import DrawLane
from Common_Tools.image_manipulation import Manipulation
from Common_Tools.line import Line

right_line = Line()
left_line = Line()
draw_lanes = DrawLane(right_line, left_line)


def process(frame, selection):
    image = crop(frame, 0, 360, 0, 854)

    man = Manipulation(selection)
    src, dst = man.get_perspective_matrix()
    transform = man.perspective_transform(image, src, dst, (854, 360))

    # Separate the V channel from the HSV image
    v = hsv(transform)
    hsv_white = set_gray_binary(v)
    hsv_white = cv2.GaussianBlur(hsv_white, (3, 3), 0)
    hsv_white = cv2.morphologyEx(hsv_white, cv2.MORPH_CLOSE, (5, 5))

    # Grayscale image
    gray = cv2.cvtColor(transform, cv2.COLOR_RGB2GRAY)
    gray = gray_equalisation(gray)

    # Blurred
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use adaptive threshold
    adap_thresh = cv2.adaptiveThreshold(gray, 225, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -15)

    # Global gradient
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Gray binary thresh
    gray_binary = set_gray_binary(gray)
    white_binary = cv2.bitwise_or(hsv_white, gray_binary)

    opening = cv2.morphologyEx(laplacian, cv2.MORPH_OPEN, (5, 5), iterations=1)
    laplacian_smooth = cv2.GaussianBlur(opening, (9, 9), 0)
    laplacian_smooth = laplacian_smooth.astype('uint8')

    combo = cv2.bitwise_and(white_binary, laplacian_smooth)
    combo = cv2.bitwise_or(combo, adap_thresh)
    combo = cv2.bitwise_or(combo, score_pixels(transform))

    get_hough_lines(combo)

    output_img = draw_lanes.find_lanes(combo)
    trans_filler = np.zeros_like(frame)
    res = draw_lanes.draw_lane(image, trans_filler, output_img, src, dst)

    result = draw_lanes.assemble_img(transform, output_img, res, combo)

    return result


def score_pixels(img):
    """
    Takes a road image and returns an image where pixel intensity maps to likelihood of it being part of the lane.
    Each pixel gets its own score, stored as pixel intensity. An intensity of zero means it is not from the lane,
    and a higher score means higher confidence of being from the lane.
    :param img: an image of a road, typically from an overhead perspective.
    :return: The score image.
    """
    # Settings to run thresholding operations on
    settings = [{'name': 'value', 'cspace': 'HSV', 'channel': 2, 'clipLimit': 6.0, 'threshold': 220}]

    # Perform binary thresholding according to each setting and combine them into one image.
    scores = np.zeros(img.shape[0:2]).astype('uint8')
    for params in settings:
        # Change color space
        color_t = getattr(cv2, 'COLOR_RGB2{}'.format(params['cspace']))
        gray = cv2.cvtColor(img, color_t)[:, :, params['channel']]

        # Normalize regions of the image using CLAHE
        clahe = cv2.createCLAHE(params['clipLimit'], tileGridSize=(8, 8))
        norm_img = clahe.apply(gray)

        # Threshold to binary
        ret, binary = cv2.threshold(norm_img, params['threshold'], 1, cv2.THRESH_BINARY)

        scores += binary

    return cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX)


def get_hough_lines(combo):

    lines = cv2.HoughLinesP(combo, rho=1, theta=1 * np.pi / 180,
                            threshold=50, minLineLength=100, maxLineGap=500)
    # Line lengths
    longest_positive_length = 0
    longest_negative_length = 0
    # Lines
    longest_positive_line = None
    longest_negative_line = None
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Draw lane lines
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if (angle < -45 and angle > -110) or (angle > 45 and angle < 110):
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
                    xn1, yn1, xn2, yn2 = longest_negative_line[0]

                    # Draw lane lines
                    if xp1 > 375 and xp2 > 375:
                        cv2.line(combo, (xp1, yp1), (xp2, yp2), (255, 255, 255), 10)
                    if xn1 < 375 and xn2 < 375:
                        cv2.line(combo, (xn1, yn1), (xn2, yn2), (255, 255, 255), 10)

    except Exception as e:
        print(e)


def crop(image, ht, hb, wt, wb):
    image = image[ht:hb, wt:wb]
    return image


def set_gray_binary(image):
    ret, thresh = cv2.threshold(image, 168, 255, cv2.THRESH_BINARY)
    return thresh


def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    result = hsv_image[:, :, 2]
    return result


def canny(image, min_val, max_val):
    result = cv2.Canny(image, min_val, max_val)
    return result


def gray_equalisation(image):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    image = clahe.apply(image)
    return image
