import cv2
import numpy as np
import time


class LaneDetection:
  def __init__(self, p):
    self.sat_thresh = p['saturation']
    self.light_thresh = p['light_yellow']
    self.light_thresh_agr = p['light_white']
    self.grad_min, self.grad_max = p['gradient']
    self.mag_thresh, self.x_thresh = p['magnitude'], p['x_thresh']
    self.hls, self.l, self.s, self.z  = None, None, None, None
    self.color_cond1, self.color_cond2 = None, None
    self.sobel_cond1, self.sobel_cond2, self.sobel_cond3 = None, None, None

  def apply(self, rgb_image):
    self.hls = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS)
    self.l = self.hls[:, :, 1]
    self.s = self.hls[:, :, 2]
    self.z = np.zeros_like(self.s)
    color_img = self.apply_color_mask()
    return color_img

  def apply_color_mask(self):
    self.color_cond1 = (self.s > self.sat_thresh) & (self.l > self.light_thresh)
    self.color_cond2 = self.l > self.light_thresh_agr
    b = self.z.copy()
    b[(self.color_cond1 | self.color_cond2)] = 255
    return b

