import cv2
import numpy as np


def process(frame, selection):
    image = crop(frame, 100, 350, 0, 854)
    # Separate the V channel from the HSV image
    v = hsv(image)
    hsv_white = set_gray_binary(v)
    hsv_white = cv2.GaussianBlur(hsv_white, (3, 3), 0)
    hsv_white = cv2.morphologyEx(hsv_white, cv2.MORPH_CLOSE, (5, 5))

    # First edge detection on whole frame
    full_canny = canny(image, 200, 200)

    # Grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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

    combo = cv2.bitwise_and(gray_binary, laplacian_smooth)
    combo = cv2.bitwise_or(combo, full_canny)
    combo = cv2.bilateralFilter(combo, 5, 150, 150)

    white_binary = set_roi(white_binary)

    return white_binary


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
