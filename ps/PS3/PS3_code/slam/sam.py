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

class Sam(SlamBase):
    def __init__(self, initial_state, alphas, state_dim=3, obs_dim=2, landmark_dim=2, action_dim=3, *args, **kwargs):
        super(Sam, self).__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.landmark_dim = landmark_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.alphas = alphas
        
        self.graph = mrob.FGraph()
        
        #first node initialization
        self.x0 = initial_state
        node0 = self.graph.add_node_pose2d(self.x0)
        
        # anchor factor 
        self.graph.add_factor_1pose_2d(self.x0, node0, )



    def predict(self, u):
        pass

    def update(self, z):
        pass
