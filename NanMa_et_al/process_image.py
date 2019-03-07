import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt

SOBEL = True

def process(frame, mtx=None, dist=None):
    image = frame
    image = crop(image)

    if mtx is not None and dist is not None:
        image = undistort(image, mtx, dist)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = gray_equalisation(image)

    img_size = (image.shape[1], image.shape[0])
    print("ims1: "+str(image.shape[1])+"  ims0: "+str(image.shape[0]))
    src = np.float32([[0, 0], [854, 0], [0, 360], [854, 360]])
    dst = np.float32([[320, 240], [550, 240], [0, 360], [854, 360]])
    image = perspective_transform(image, src, dst, img_size)

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


def perspective_transform(image, src, dst, img_size):
    matrix = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, matrix, img_size)
    return warped