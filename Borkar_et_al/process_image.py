import cv2
import numpy as np
from Common_Tools.image_manipulation import Manipulation
from Common_Tools.draw_lane import DrawLane
from Common_Tools.line import Line
import math

right_line = Line()
left_line = Line()
draw_lanes = DrawLane(right_line, left_line)


def process(image, selection):
    image = crop(image, 0, 360, 0, 854)
    # Get inverse perspective transform
    man = Manipulation(selection)
    src, dst = man.get_perspective_matrix()
    transform = man.perspective_transform(image, src, dst, (854, 360))
    #transform = cv2.GaussianBlur(transform, (5, 5), 0)

    # Convert to grayscale
    gray = cv2.cvtColor(transform, cv2.COLOR_RGB2GRAY)
    # Use adaptive threshold
    adap_thresh = cv2.adaptiveThreshold(gray, 225, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -15)

    #Separate vertical pixels
    vertical = adap_thresh
    rows = vertical.shape[0]
    # v_pix_size is the number of rows to be considered a valid vertical blob
    v_pix_size = rows / 20
    v_pix_size = int(v_pix_size)
    v_struc = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_pix_size))
    # Apply morphology operations
    vertical = cv2.erode(vertical, v_struc)
    adap_thresh = cv2.dilate(vertical, v_struc)


    #adap_thresh, lines = get_hough_lines(adap_thresh)
    # Sampling Columns
    #col_image = draw_sampling_columns(adap_thresh)

    output_img = draw_lanes.find_lanes(adap_thresh)
    trans_filler = np.zeros_like(image)
    res = draw_lanes.draw_lane(image, trans_filler, output_img, src, dst)

    result = draw_lanes.assemble_img(transform, output_img, res, adap_thresh)

    return result


def get_hough_lines(image):
    longest_negative_line = None
    longest_positive_line = None
    longest_positive_length = 0
    longest_negative_length = 0
    lines = cv2.HoughLinesP(image, 1, 1 * np.pi / 180, 150, None, minLineLength=150, maxLineGap=250)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if (angle < -60 and angle > -100) or (angle > 60 and angle < 100):
                    # get the longest positive and negative line
                    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if angle > 0:
                        if length > longest_positive_length:
                            print("Pos: " + str(angle))
                            longest_positive_length = length
                            longest_positive_line = line
                    elif angle < 0:
                        if length > longest_negative_length:
                            print("Neg: " + str(angle))
                            longest_negative_length = length
                            longest_negative_line = line

                if longest_positive_length != 0 and longest_negative_length != 0:
                    # Positive Line
                    xp1, yp1, xp2, yp2 = longest_positive_line[0]
                    xn1, yn1, xn2, yn2 = longest_negative_line[0]

                    # Draw lane lines
                    if xp1 > 375 and xp2 > 375:
                        cv2.line(image, (xp1, yp1), (xp2, yp2), (255, 255, 255), 10)
                    if xn1 < 375 and xn2 < 375:
                        cv2.line(image, (xn1, yn1), (xn2, yn2), (255, 255, 255), 10)
    return image, lines


def draw_sampling_columns(image):
    count = 0
    loc = 36

    while count < 9:
        cv2.line(image, (0, loc), (854, loc), (255, 255, 255), 1)
        loc += 36
        count += 1

    return image


def crop(image, ht, hb, wt, wb):
    image = image[ht:hb, wt:wb]
    return image
