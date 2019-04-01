import numpy as np
from Common_Tools.kalman_filter import KalFilter


class Window:
    def __init__(self, img_shape, start_x_pos, win_num, margin):

        # Number of the window in the stack
        self.win_num = win_num

        # Image shape
        self.img_h = img_shape[0]
        self.img_w = img_shape[1]

        # Window margin
        self.margin = margin
        self.minpix = 50

        # The number of sliding windows
        self.nwindows = 8
        # Set height of windows
        self.window_height = np.int(self.img_h / self.nwindows)

        # Kalman filter
        self.kf = KalFilter(pos_init=start_x_pos, meas_variance=0)
        self.x_measured = start_x_pos
        self.x_filtered = start_x_pos
        self.y_begin = self.img_h - (win_num + 1) * self.window_height
        self.y = self.y_begin + self.window_height / 2.0
        self.detected = False

    def freeze(self):
        self.kf.grow_uncertainty(1)

    def update(self, nonzerox, nonzeroy, min_log_likelihood=-20):

        win_y_low = self.img_h - (self.win_num + 1) * self.window_height
        win_y_high = self.img_h - self.win_num * self.window_height
        win_x_low = self.x_filtered - self.margin
        win_x_high = self.x_filtered + self.margin

        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (
                nonzerox < win_x_high)).nonzero()[0]

        good_inds = good_inds.astype(int)

        if len(good_inds) > self.minpix:
            self.detected = True
            self.x_measured = np.int(np.mean(nonzerox[good_inds]))
        else:
            self.detected = False
            self.freeze()
            return good_inds

        self.kf.update(self.x_measured)
        self.x_filtered = self.kf.get_position()

        return good_inds

    def get_win_pos(self):
        return np.int(self.x_filtered), np.int(self.y)

    def get_x(self):
        return self.x_filtered
