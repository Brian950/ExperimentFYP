from collections import deque
import numpy as np

class Line():
    def __init__(self, maxSamples=4):
        self.maxSamples = maxSamples
        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=self.maxSamples)
        # Polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # Polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # Average x values of the fitted line over the last n iterations
        self.bestx = None
        # Was the line detected in the last iteration?
        self.detected = False
        # Radius of curvature of the line in some units
        self.radius_of_curvature = None
        # Distance in meters of vehicle center from the line
        self.line_base_pos = None

    def update_lane(self, ally, allx):
        # Updates lanes on every new frame
        # Mean x value
        self.bestx = np.mean(allx, axis=0)
        # Fit 2nd order polynomial
        new_fit = np.polyfit(ally, allx, 2)
        # Update current fit
        self.current_fit = new_fit
        # Add the new fit to the queue
        self.recent_xfitted.append(self.current_fit)
        # Use the queue mean as the best fit
        self.best_fit = np.mean(self.recent_xfitted, axis=0)
        # meters per pixel in y dimension
        ym_per_pix = 30 / 720
        # meters per pixel in x dimension
        xm_per_pix = 3.7 / 700
        # Calculate radius of curvature
        fit_cr = np.polyfit(ally * ym_per_pix, allx * xm_per_pix, 2)
        y_eval = np.max(ally)
        self.radius_of_curvature = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_cr[0])