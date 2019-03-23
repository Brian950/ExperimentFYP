from typing import List, Tuple
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from Common_Tools.lane_filter import LaneFilter


class Window:
    def __init__(self, level, window_shape, img_shape, x_init, max_frozen_dur):
        """
        Tracks a window as used for selecting lane lines in an image.
        :param level: Level of the window, as counted from the bottom of the image up.
        :param window_shape: (height, width) of the window in pixels.
        :param img_shape: (height, width) of the image the window resides in.
        :param x_init: Initial x position of the window.
        :param max_frozen_dur: The maximum amount of frames a window can continue to be used when frozen (eg when not
        found or when measurements are uncertain).
        """
        if window_shape[1] % 2 == 0:
            raise Exception("width must be odd")
        # Image info
        self.img_h = img_shape[0]
        self.img_w = img_shape[1]

        # Window shape
        self.height = window_shape[0]
        self.width = window_shape[1]
        self.y_begin = self.img_h - (level + 1) * self.height  # top row of pixels for window
        self.y_end = self.y_begin + self.height  # one past the bottom row of pixels for window

        # Window position
        self.x_filtered = x_init
        self.y = self.y_begin + self.height / 2.0
        self.level = level

        # Detection info
        self.filter = LaneFilter(x_pos=x_init)
        self.x_measured = None
        self.frozen = False
        self.detected = False
        self.max_frozen_dur = max_frozen_dur
        self.frozen_dur = max_frozen_dur + 1
        self.undrop_buffer = 1  # Number of calls to unfreeze() needed to go from dropped back to normal.

    def x_begin(self, param='x_filtered'):
        """
        The leftmost position of the window, relative to the last filtered position or measurement.
        :param param: Whether to use the 'x_filtered' or 'x_measured' position.
        """
        self.check_x_param(param)
        x = getattr(self, param)
        return int(max(0, x - self.width // 2))

    def x_end(self, param='x_filtered'):
        """
        One past the rightmost position of the window, relative to the last filtered position or measurement.
        :param param: Whether to use the 'x_filtered' or 'x_measured' position.
        """
        self.check_x_param(param)
        x = getattr(self, param)
        return int(min(x + self.width // 2, self.img_w))

    def area(self):
        """Area of the window."""
        return self.height * self.width

    def freeze(self):
        """Marks the window as frozen, drops it if it's been frozen for too long, and increases filter uncertainty."""
        self.frozen = True
        self.frozen_dur += 1
        self.filter.grow_uncertainty(1)

    def unfreeze(self):
        """Marks the window as not frozen and not dropped, reduces frozen counter by 1."""
        # Reduce frozen duration to max (plus some buffer)
        self.frozen_dur = min(self.frozen_dur, self.max_frozen_dur + 1 + self.undrop_buffer)
        self.frozen_dur -= 1
        self.frozen_dur = max(0, self.frozen_dur)

        # Change states
        self.frozen = False

    @property
    def dropped(self):
        return self.frozen_dur > self.max_frozen_dur

    def update(self, score_img, x_search_range, min_log_likelihood=-40):
        """
        Given a score image and the x search bounds, updates the window position to the likely position of the lane.
        If the measurement is deemed suspect for some reason, the update will be rejected and the window will be
        'frozen', causing it to stay in place. If the window is frozen beyond its  `max_frozen_dur` then it will be
        dropped entirely until a non-suspect measurement is made.
        The window only searches within its y range defined at initialization.
        :param score_img: A score image, where pixel intensity represents where the lane most likely is.
        :param x_search_range: The (x_begin, x_end) range the window should search between in the score image.
        :param min_log_likelihood: The minimum log likelihood allowed for a measurement before it is rejected.
        """
        assert score_img.shape[0] == self.img_h and \
               score_img.shape[1] == self.img_w, 'Window not parametrized for this score_img size'

        # Apply a column-wise gaussian filter to score the x-positions in this window's search region
        x_search_range = (max(0, int(x_search_range[0])), min(int(x_search_range[1]), self.img_w))
        x_offset = x_search_range[0]
        search_region = score_img[self.y_begin: self.y_end, x_offset: x_search_range[1]]
        column_scores = gaussian_filter(np.sum(search_region, axis=0), sigma=self.width / 3, truncate=3.0)

        if max(column_scores) != 0:
            self.detected = True
            # Update measurement
            self.x_measured = np.argmax(column_scores) + x_offset
            window_magnitude = \
                np.sum(column_scores[self.x_begin('x_measured') - x_offset: self.x_end('x_measured') - x_offset])
            noise_magnitude = np.sum(column_scores) - window_magnitude
            signal_noise_ratio = \
                window_magnitude / (window_magnitude + noise_magnitude) if window_magnitude is not 0 else 0

            # Filter measurement and set position
            if signal_noise_ratio < 0.6 or self.filter.loglikelihood(self.x_measured) < min_log_likelihood:
                # Suspect / bad measurement, don't update filter/position
                self.freeze()
                return
            self.unfreeze()
            self.filter.update(self.x_measured)
            self.x_filtered = self.filter.get_position()

        else:
            # No signal in search region
            self.detected = False
            self.freeze()

    def pos_xy(self, param: str = 'x_filtered') -> Tuple[float, float]:
        """Returns the (x, y) position of this window."""
        self.check_x_param(param)
        return getattr(self, param), self.y

    def check_x_param(self, param):
        assert param == 'x_filtered' or param == 'x_measured', "Invalid position parameter. `param` must be " \
                                                               "'x_filtered' or 'x_measured' "