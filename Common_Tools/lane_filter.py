import cv2
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter, logpdf, dot


class LaneFilter:

    def __init__(self, x_pos=1.0, variance=1, m_variance=50, uncertainty=2**30):
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        # State transition matrix
        self.kf.F = np.array([[1., 1],
                              [0., 0.5]])
        # Measurement
        self.kf.H = np.array([[1., 0.]])
        # Covariance matrix
        self.kf.P = np.eye(self.kf.dim_x) * uncertainty
        # State estimate
        self.kf.x = np.array([x_pos, 0])
        # Noise
        self.kf.R = np.array([[m_variance]])
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=variance)

    def update(self, x_pos):
        self.kf.predict()
        self.kf.update(x_pos)

    def get_x_pos(self):
        return self.kf.x[0]

    def update_uncertainty(self, var):
        for x in range(var):
            self.kf.P = self.kf._alpha_sq * dot(self.kf.F, self.kf.P).dot(self.kf.F.T)

    def log_score(self, pos):
        # Log likelihood
        self.kf.S = dot(self.kf.H, self.kf.P).dot(self.kf.H.T) + self.kf.R
        return logpdf(pos, np.dot(self.kf.H, self.kf.x), self.kf.S)
