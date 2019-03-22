import cv2
import glob
import numpy as np
from NanMa_et_al.hsv_mask import HSVMask
from Common_Tools.line import Line
from Common_Tools.image_manipulation import Manipulation
from Common_Tools.draw_lane import DrawLane

SOBEL = True
img_size = (854, 360)

lane_params = {'saturation': 100, 'light_yellow': 60, 'light_white': 195,
               'gradient': (0.7, 1.5), 'x_thresh': 20, 'magnitude': 40}
lane_finder = HSVMask(lane_params)

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

    man = Manipulation(selection)
    src, dst = man.get_perspective_matrix()

    transform = man.perspective_transform(view, src, dst, img_size)

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

    draw_lanes = DrawLane(right_line, left_line)
    output_img = draw_lanes.find_lanes(combo_image)
    trans_filler = np.zeros_like(frame)
    res = draw_lanes.draw_lane(view, trans_filler, output_img, src, dst)

    result = draw_lanes.assemble_img(transform, output_img, res, combo_image)

    return result


def set_gray_binary(image):
    ret, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    return thresh


def crop(image, ht, hb, wt, wb):
    image = image[ht:hb, wt:wb]
    return image


def gaussian_blur(img, x, y):
    gauss_img = cv2.GaussianBlur(img, (x, y), 0)
    return gauss_img


def undistort(frame, mtx, dist):
    frame = cv2.undistort(frame, mtx, dist, None, mtx)
    return frame


def gray_equalisation(image):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    image = clahe.apply(image)
    return image


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
