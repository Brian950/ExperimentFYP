import numpy as np
import cv2
import math


class Manipulation:

    def __init__(self, selection):
        self.selection = selection

    def get_perspective_matrix(self):

        if self.selection == "highway.mp4":
            src = np.float32([[0, 0], [854, 0], [0, 360], [854, 360]])
            dst = np.float32([[320, 240], [550, 240], [0, 360], [854, 360]])
            return src, dst
        elif self.selection == "highway_sunlight.mp4":
            src = np.float32([[0, 0], [854, 0], [0, 360], [854, 360]])
            dst = np.float32([[280, 200], [500, 200], [0, 360], [854, 360]])
            return src, dst
        elif self.selection == "highway_night.mp4":
            src = np.float32([[0, 0], [854, 0], [0, 360], [854, 360]])
            dst = np.float32([[380, 280], [600, 280], [0, 360], [854, 360]])
            return src, dst
        else:
            src = np.float32([[0, 0], [854, 0], [0, 360], [854, 360]])
            dst = np.float32([[240, 200], [500, 200], [0, 360], [854, 360]])
            return src, dst
