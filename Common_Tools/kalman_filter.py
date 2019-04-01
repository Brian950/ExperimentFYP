import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter, logpdf, dot


class KalFilter:

    def __init__(self, pos_init=0.0, meas_variance=50, process_variance=1, uncertainty_init=1):

        self.kf = KalmanFilter(dim_x=2, dim_z=1)

        # State transition function
        self.kf.F = np.array([[1., 1],
                              [0., 1]])

        # Measurement function
        self.kf.H = np.array([[1., 0.]])

        # Initial state estimate
        self.kf.x = np.array([pos_init, 0])

        # Initial Covariance matrix
        self.kf.P = np.eye(self.kf.dim_x) * uncertainty_init

        # Measurement noise
        self.kf.R = np.array([[meas_variance]])

        # Process noise
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=process_variance)

    def update(self, pos):
        self.kf.predict()
        self.kf.update(pos)

    def grow_uncertainty(self, r):
        for i in range(r):
            self.kf.P = self.kf._alpha_sq * dot(dot(self.kf.F, self.kf.P), self.kf.F.T) + self.kf.Q

    def loglikelihood(self, pos):
        self.kf.S = dot(dot(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R
        return logpdf(pos, np.dot(self.kf.H, self.kf.x), self.kf.S)

    def get_position(self):
        return self.kf.x[0]
