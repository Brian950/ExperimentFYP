from collections import deque
import numpy as np


class Line:

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
        #new_fit = self.ransac_polyfit(allx, ally, k=2)
        # Update current fit
        self.current_fit = new_fit
        # Add the new fit to the queue
        self.recent_xfitted.append(self.current_fit)
        # Use the queue mean as the best fit
        self.best_fit = np.mean(self.recent_xfitted, axis=0)
        #self.best_fit = self.current_fit

    def ransac_polyfit(self, x, y, order=3, n=20, k=100, t=0.1, d=50, f=0.8):
        # https://gist.github.com/geohot/9743ad59598daf61155bf0d43a10838c

        # n – minimum number of data points required to fit the model
        # k – maximum number of iterations allowed in the algorithm
        # t – threshold value to determine when a data point fits a model
        # d – number of close data points required to assert that a model fits well to data
        # f – fraction of close data points required

        besterr = np.inf
        bestfit = None
        for kk in range(k):
            maybeinliers = np.random.randint(len(x), size=n)
            maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
            alsoinliers = np.abs(np.polyval(maybemodel, x) - y) < t
            if sum(alsoinliers) > d and sum(alsoinliers) > len(x) * f:
                bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], order)
                thiserr = np.sum(np.abs(np.polyval(bettermodel,
                                                   x[alsoinliers]) - y[alsoinliers]))
                if thiserr < besterr:
                    bestfit = bettermodel
                    besterr = thiserr

        return bestfit