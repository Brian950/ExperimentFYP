from collections import deque
import numpy as np


class Line:

    def __init__(self, samples=4):
        self.maxSamples = samples
        # x values of the last 'samples'
        self.recent_xfitted = deque(maxlen=self.maxSamples)
        self.current_fit = [np.array([False])]
        # Average of the lest samples
        self.best_fit = None
        self.bestx = None
        # If the lien is detected successfully
        self.detected = False

    def update_lane(self, ally, allx):
        # Mean x value
        self.bestx = np.mean(allx, axis=0)

        new_fit = np.polyfit(ally, allx, 2)
        self.current_fit = new_fit
        # Queue of line samples
        self.recent_xfitted.append(self.current_fit)
        # Average of queue gets the best fit
        self.best_fit = np.mean(self.recent_xfitted, axis=0)

    def reset(self):
        self.recent_xfitted.clear()
        self.current_fit = [np.array([False])]
        self.best_fit = None
        self.bestx = None
        self.detected = False
