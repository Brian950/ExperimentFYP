import numpy as np
import cv2
import glob
import math


class Manipulation:

    def __init__(self, selection):
        self.selection = selection

    def get_perspective_matrix(self):

        if self.selection == "highway.mp4":
            src = np.float32([[0, 0], [854, 0], [0, 360], [854, 360]])
            dst = np.float32([[320, 240], [550, 240], [0, 360], [854, 360]])
            return src, dst
        elif self.selection == "highway2.mp4":
            src = np.float32([[0, 0], [854, 0], [0, 360], [854, 360]])
            dst = np.float32([[340, 300], [520, 300], [0, 360], [854, 360]])
            return src, dst
        elif self.selection == "highway_sunlight.mp4":
            src = np.float32([[0, 0], [854, 0], [0, 360], [854, 360]])
            dst = np.float32([[280, 200], [500, 200], [0, 360], [854, 360]])
            return src, dst
        elif self.selection == "highway_night.mp4":
            src = np.float32([[0, 0], [854, 0], [0, 360], [854, 360]])
            dst = np.float32([[340, 280], [620, 280], [0, 360], [854, 360]])
            return src, dst
        else:
            src = np.float32([[0, 0], [854, 0], [0, 360], [854, 360]])
            dst = np.float32([[320, 240], [550, 240], [0, 360], [854, 360]])
            return src, dst

    def perspective_transform(self, image, src, dst, img_size):
        matrix = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(image, matrix, img_size)
        return warped

    # Calibration method modified from https://opencv-python-tutroals.readthedocs.io
    def calibrate_camera(self, directory, nx, ny, img_size):
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        images = glob.glob(directory + '*.jpg')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        return mtx, dist