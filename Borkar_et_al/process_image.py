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

    # Convert to grayscale
    gray = cv2.cvtColor(transform, cv2.COLOR_RGB2GRAY)
    # Use adaptive threshold
    adap_thresh = cv2.adaptiveThreshold(gray, 225, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -15)

    # Image halves
    left_side = crop(adap_thresh, 0, 360, 200, 500)
    right_side = crop(adap_thresh, 0, 360, 450, 800)

    # Uses the left & right sides to draw on adap_thresh
    get_hough_lines(left_side, adap_thresh, left=True)
    get_hough_lines(right_side, adap_thresh, right=True)

    output_img = draw_lanes.find_lanes(adap_thresh)
    trans_filler = np.zeros_like(image)
    res = draw_lanes.draw_lane(image, trans_filler, output_img, src, dst)

    result = assemble_img(transform, output_img, res, adap_thresh,
                          cv2.cvtColor(left_side, cv2.COLOR_GRAY2RGB),
                          cv2.cvtColor(right_side, cv2.COLOR_GRAY2RGB))

    return result


def get_hough_lines(split_image, whole_image, left=False, right=False):

    if not left and not right:
        return

    lines = cv2.HoughLinesP(split_image, rho=1, theta=np.pi / 180,
                            threshold=50, minLineLength=75, maxLineGap=200)
    # Line lengths
    longest_positive_length = 0
    longest_negative_length = 0
    # Lines
    longest_positive_line = None
    longest_negative_line = None
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if left:
                    if -45 > angle > -90:
                        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        if length > longest_negative_length:
                            longest_negative_length = length
                            longest_negative_line = line
                elif right:
                    if 45 < angle < 90:
                        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        if length > longest_positive_length:
                            longest_positive_length = length
                            longest_positive_line = line
                else:
                    return

        if longest_positive_length != 0:
            # Positive Line
            xp1, yp1, xp2, yp2 = longest_positive_line[0]
            cv2.line(whole_image, (xp1+450, yp1), (xp2+450, yp2), (255, 255, 255), 10)
        if longest_negative_length != 0:
            # Negative Line
            xn1, yn1, xn2, yn2 = longest_negative_line[0]
            cv2.line(whole_image, (xn1+200, yn1), (xn2+200, yn2), (255, 255, 255), 10)

    except Exception as e:
        print(e)


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


def assemble_img(warped, polynomial_img, lane_img, adap, left_side, right_side):
    # Output image
    img_out = np.zeros((740, 1290, 3), dtype=np.uint8)
    if lane_img is not None:
        img_out[0:360, 0:854, :] = lane_img

    combo_image = cv2.cvtColor(adap, cv2.COLOR_GRAY2RGB)
    font_size = 1
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Perspective transform
    img_out[0:240, 865:1285, :] = cv2.resize(warped, (420, 240))
    boxsize, _ = cv2.getTextSize("Transformed", font, font_size, thickness)
    cv2.putText(img_out, "Transformed", (int(1090 - boxsize[0] / 2), 40), font, font_size, (50, 200, 255),
                thickness, lineType=cv2.LINE_AA)

    # Threshold
    img_out[250:490, 865:1285, :] = cv2.resize(combo_image, (420, 240))
    boxsize, _ = cv2.getTextSize("Threshold", font, font_size, thickness)
    cv2.putText(img_out, "Threshold", (int(1090 - boxsize[0] / 2), 280), font, font_size, (200, 50, 255),
                thickness, lineType=cv2.LINE_AA)

    # Lane lines
    img_out[500: 740, 865:1285, :] = cv2.resize(polynomial_img * 255, (420, 240))
    boxsize, _ = cv2.getTextSize("Detected Lane", font, font_size, thickness)
    cv2.putText(img_out, "Detected Lane", (int(1090 - boxsize[0] / 2), 520), font, font_size, (255, 255, 255),
                thickness, lineType=cv2.LINE_AA)

    # HSV
    img_out[500: 740, 440:860, :] = cv2.resize(right_side, (420, 240))
    boxsize, _ = cv2.getTextSize("Right Side", font, font_size, thickness)
    cv2.putText(img_out, "Right Side", (int(650 - boxsize[0] / 2), 530), font, font_size, (200, 255, 50),
                thickness, lineType=cv2.LINE_AA)

    # Laplacian
    img_out[500: 740, 10:430, :] = cv2.resize(left_side, (420, 240))
    boxsize, _ = cv2.getTextSize("Left Side", font, font_size, thickness)
    cv2.putText(img_out, "Left Side", (int(230 - boxsize[0] / 2), 530), font, font_size, (50, 255, 200),
                thickness, lineType=cv2.LINE_AA)

    return img_out