"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle

from field_map import FieldMap


class EKF(LocalizationFilter):
    def get_jacobian_G(self, state, motion):
        """
        state: state of the robot ([x, y, theta] need only theta)
        motion: motion command ([delta_rot1, delta_tran, delta_rot2])
        return: G_t
        """

        theta = state[2]
        delta_rot1, delta_tran = motion[0], motion[1]

        return np.array([
            [1, 0, -delta_tran * np.sin(theta + delta_rot1)],
            [0, 1, delta_tran * np.cos(theta + delta_rot1)],
            [0, 0, 1]
            ])
   
    def get_jacobian_V(self, state, motion):
        """
        state: state of the robot ([x, y, theta] need only theta)
        motion: motion command ([delta_rot1, delta_tran, delta_rot2])
        return: V_t
        """

        theta = state[2]
        delta_rot1, delta_tran = motion[0], motion[1]

        return np.array([
            [-delta_tran * np.sin(theta + delta_rot1), np.cos(theta + delta_rot1), 0],
            [delta_tran * np.cos(theta + delta_rot1), np.sin(theta + delta_rot1), 0],
            [1, 0, 1]
            ])

    def get_jacobian_H(self, state, lm_id):
        """
        state: state of the robot ([x, y, theta] need only x, y)
        lm_id: landmark id 
        return: H_t
        """

        x, y = state[0], state[1]
        lm_id = int(lm_id)
        field_map = FieldMap()

        # dx = field_map.landmarks_poses_x[lm_id] - x
        # dy = field_map.landmarks_poses_y[lm_id] - y
        dx = self._field_map.landmarks_poses_x[lm_id] - x 
        dy = self._field_map.landmarks_poses_y[lm_id] - y
        q = dy**2 + dx**2

        return np.array([[dy/q, -dx/q, -1]])   

    def predict(self, u):
        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task
        R_t = get_motion_noise_covariance(u, self._alphas)
        
        G_t = self.get_jacobian_G(self.mu, u)
        V_t = self.get_jacobian_V(self.mu, u)

        mu = get_prediction(np.ravel(self.mu), u)
        Sigma = G_t @ self._state.Sigma @ G_t.T + V_t @ R_t @ V_t.T

        self._state_bar.mu = mu[np.newaxis].T
        self._state_bar.Sigma = Sigma

    def update(self, z):
        # TODO implement correction step
        phi, lm_id = z[0], int(z[1])

        H_t_i = self.get_jacobian_H(self.mu_bar, lm_id)
        S_t_i = H_t_i @ self._state_bar.Sigma @ H_t_i.T + self._Q
        K_t_i = self._state_bar.Sigma @ H_t_i.T / S_t_i
        
        self._state_bar.mu = self._state_bar.mu + K_t_i * wrap_angle(phi - get_expected_observation(self.mu_bar, lm_id)[0])
        self._state_bar.Sigma = (np.eye(3) - K_t_i @ H_t_i) @ self._state_bar.Sigma

        self._state.mu = self._state_bar.mu
        self._state.Sigma = self._state_bar.Sigma
