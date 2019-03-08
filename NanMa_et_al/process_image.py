import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt

SOBEL = True

src = np.float32([[0, 0], [854, 0], [0, 360], [854, 360]])
dst = np.float32([[320, 230], [550, 230], [0, 360], [854, 360]])
img_size = (854, 360)

sharp_kernel = np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]])


def process(frame, mtx=None, dist=None):
    image = frame
    image = crop(image)

    if mtx is not None and dist is not None:
        view = undistort(image, mtx, dist)

    transform = perspective_transform(view, src, dst, img_size)

    gray = cv2.cvtColor(transform, cv2.COLOR_RGB2GRAY)
    eq_image = gray_equalisation2(gray)

    sharp = sharpen(eq_image)

    sobel_image = sobel(sharp, 3)

    canny_image = canny(sharp, 50, 200)

    binary_image = set_gray_binary(sharp)

    return binary_image


def set_gray_binary(image):
    ret, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    return thresh


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


def crop(image):
    image = image[0:360, 0:854]
    return image


# Calibration method modified from https://opencv-python-tutroals.readthedocs.io
def calibrate_camera(directory, nx, ny, img_size):
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

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


'''
This function works by first finding the minimum histogram value excluding zero
by masking the array of pixels (cdf_m). The histogram equalisation formula is
then applied to the masked array and then the masked array is used as a lookup
to transform the original image. Noticeable 5-6 fps drop using this algorithm.
'''


def gray_equalisation(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    cdf = hist.cumsum()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    image = cdf[image]
    return image


def gray_equalisation2(image):
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
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