"""
Gonzalo Ferrer
g.ferrer@skoltech.ru
28-Feb-2021
"""

import numpy as np
import mrob
from scipy.linalg import inv
from slam.slamBase import SlamBase
from tools.task import get_motion_noise_covariance
from tools.jacobian import state_jacobian


class Sam(SlamBase):
    def __init__(self, initial_state, alphas, state_dim=3, obs_dim=2, landmark_dim=2, action_dim=3, *args, **kwargs):
        super(Sam, self).__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.landmark_dim = landmark_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.alphas = alphas
        self.lm_index = dict()

        self.x =  np.squeeze(initial_state.mu)       

        self.graph = mrob.FGraph()

        self.x_index = self.graph.add_node_pose_2d(self.x)
        self.W = inv(initial_state.Sigma) # information observation matrix
        self.graph.add_factor_1pose_2d(self.x, self.x_index, self.W)
 
        # self.graph.print(True)



    def predict(self, u):
        prev_state = self.graph.get_estimated_state()
        print('State values before adding the odometry factor: ', *prev_state)

        x_new = self.graph.add_node_pose_2d(np.zeros(3))
        G, V = state_jacobian(self.x, u)
        M = get_motion_noise_covariance(u, self.alphas)

        W_u = inv(V @ M @ V.T)
        self.graph.add_factor_2poses_2d_odom(u, self.x_index, x_new, W_u)
        curr_state = self.graph.get_estimated_state()
        print('State values after adding the odometry factor: ', *curr_state)

        self.x_index = x_new
        self.x = np.squeeze(curr_state[-1])

    def update(self, z):
        W_z = inv(self.Q)
        nodes = self.graph.get_estimated_state()
        for observations in z:
            is_new = False
            if observations[-1] in self.lm_index:
                lm_i = self.lm_index[observations[-1]]
            else:
                is_new = True
                lm_i = self.graph.add_node_landmark_2d(np.zeros(2))

            self.lm_index[observations[-1]] = int(lm_i)
            self.graph.add_factor_1pose_1landmark_2d(
                observations[:-1], 
                self.x_index, 
                self.lm_index[observations[-1]], 
                W_z, 
                initializeLandmark=is_new
                )
        self.graph.print(True)

    def solve(self, gn=True):
        if gn:
            self.graph.solve()
        else:
            self.graph.solve(method=mrob.LM)
        self.graph.print(True)