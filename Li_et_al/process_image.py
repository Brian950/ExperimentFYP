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
    transform = perspective_transform(image, src, dst, (854, 360))

    # Separate the V channel from the HSV image
    v = hsv(transform)
    hsv_white = set_gray_binary(v)
    hsv_white = cv2.GaussianBlur(hsv_white, (3, 3), 0)
    hsv_white = cv2.morphologyEx(hsv_white, cv2.MORPH_CLOSE, (5, 5))

    # First edge detection on whole frame
    full_canny = canny(transform, 200, 200)

    # Grayscale image
    gray = cv2.cvtColor(transform, cv2.COLOR_RGB2GRAY)
    #gray = gray_equalisation(gray)

    # Blurred
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Global gradient
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Gray binary thresh
    gray_binary = set_gray_binary(gray)
    white_binary = cv2.bitwise_or(hsv_white, gray_binary)

    opening = cv2.morphologyEx(laplacian, cv2.MORPH_OPEN, (5, 5), iterations=1)
    laplacian_smooth = cv2.GaussianBlur(opening, (9, 9), 0)
    laplacian_smooth = laplacian_smooth.astype('uint8')

    combo = cv2.bitwise_and(white_binary, laplacian_smooth)
    combo = cv2.bitwise_or(combo, full_canny)
    # Noise removal
    combo = cv2.bilateralFilter(combo, 5, 150, 150)

    left_combo = crop(combo, 0, 360, 0, 427)
    right_combo = crop(combo, 0, 360, 427, 854)
    lines = cv2.HoughLinesP(combo, rho=1, theta=1 * np.pi / 180,
                            threshold=40, minLineLength=100, maxLineGap=500)

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
                if (angle < -60 and angle > -90) or (angle > 60 and angle < 90):
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
                        cv2.line(combo, (xp1, yp1), (xp2, yp2), (255, 255, 255), 10)
                    if xn1 < 375 and xn2 < 375:
                        cv2.line(combo, (xn1, yn1), (xn2, yn2), (255, 255, 255), 10)

    except Exception as e:
        print(e)

    output_img = draw_lanes.find_lanes(combo)
    trans_filler = np.zeros_like(frame)
    res = draw_lanes.draw_lane(image, trans_filler, output_img, src, dst)

    result = draw_lanes.assemble_img(transform, output_img, res, combo)

    return result


def perspective_transform(image, src, dst, img_size):
    matrix = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, matrix, img_size)
    return warped


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
