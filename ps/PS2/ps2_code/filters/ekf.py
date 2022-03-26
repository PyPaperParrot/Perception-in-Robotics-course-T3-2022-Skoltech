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
        state: state of the robot ([x, y, theta])
        motion: motion command ([drot1, dtran, drot2])
        return: Gt
        """

        assert isinstance(state, np.ndarray)
        assert isinstance(motion, np.ndarray)
        assert state.shape == (3,)
        assert motion.shape == (3,)

        x, y, theta = state
        drot1, dtran, drot2 = motion

        return np.array([
            [1, 0, -dtran * np.sin(theta + drot1)],
            [0, 1, dtran * np.cos(theta + drot1)],
            [0, 0, 1]
            ])
   
    def get_jacobian_V(self, state, motion):
        """
        state: state of the robot ([x, y, theta])
        motion: motion command ([drot1, dtran, drot2])
        return: Vt
        """

        assert isinstance(state, np.ndarray)
        assert isinstance(motion, np.ndarray)
        assert state.shape == (3,)
        assert motion.shape == (3,)

        x, y, theta = state
        drot1, dtran, drot2 = motion

        return np.array([
            [-dtran * np.sin(theta + drot1), np.cos(theta + drot1), 0],
            [dtran * np.cos(theta + drot1), np.sin(theta + drot1), 0],
            [1, 0, 1]
            ])

    def get_jacobian_H(self, state, lm_id):
        """
        :param state: The current state of the robot (format: [x, y, theta]).
        :param lm_id: The landmark id indexing into the landmarks list in the field map.
        :return: Ht.
        """

        assert isinstance(state, np.ndarray)
        assert state.shape == (3,)

        x, y = state[:2]
        lm_id = int(lm_id)
        field_map = FieldMap()

        dx = field_map.landmarks_poses_x[lm_id] - x
        dy = field_map.landmarks_poses_y[lm_id] - y
        q = dx**2 + dy**2

        return np.array([dy/q, -dx/q, -1])   

    def predict(self, u):
        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task
        self._state_bar.mu = self.mu[np.newaxis].T
        self._state_bar.Sigma = self.Sigma

    def update(self, z):
        # TODO implement correction step
        self._state.mu = self._state_bar.mu
        self._state.Sigma = self._state_bar.Sigma
