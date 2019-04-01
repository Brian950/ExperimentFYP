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

    get_hough_lines(combo)

    output_img = draw_lanes.find_lanes(combo)
    trans_filler = np.zeros_like(frame)
    res = draw_lanes.draw_lane(image, trans_filler, output_img, src, dst)

    result = assemble_img(transform, output_img,
                          res, combo, cv2.cvtColor(laplacian_smooth, cv2.COLOR_GRAY2RGB),
                          cv2.cvtColor(hsv_white, cv2.COLOR_GRAY2RGB))


    return result


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
                # Get line angle
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if (angle < -60 and angle > -90) or (angle > 60 and angle < 90):
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


def assemble_img(warped, polynomial_img, lane_img, combo_image, laplacian, gray_eq):
    # Output image
    img_out = np.zeros((740, 1290, 3), dtype=np.uint8)
    if lane_img is not None:
        img_out[0:360, 0:854, :] = lane_img

    combo_image = cv2.cvtColor(combo_image, cv2.COLOR_GRAY2RGB)
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
    img_out[500: 740, 440:860, :] = cv2.resize(gray_eq, (420, 240))
    boxsize, _ = cv2.getTextSize("HSV Separation ", font, font_size, thickness)
    cv2.putText(img_out, "HSV Separation", (int(650 - boxsize[0] / 2), 530), font, font_size, (200, 255, 50),
                thickness, lineType=cv2.LINE_AA)

    # Laplacian
    img_out[500: 740, 10:430, :] = cv2.resize(laplacian, (420, 240))
    boxsize, _ = cv2.getTextSize("Laplacian", font, font_size, thickness)
    cv2.putText(img_out, "Laplacian", (int(230 - boxsize[0] / 2), 530), font, font_size, (50, 255, 200),
                thickness, lineType=cv2.LINE_AA)

    return img_out