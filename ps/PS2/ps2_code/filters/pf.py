"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.objects import Gaussian
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle


class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, bearing_std, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, bearing_std)
        # TODO add here specific class variables for the PF

        self.num_particles = num_particles
        self.particles = np.ones((self.num_particles, 3)) * initial_state.mu[:, 0] 
        self.bearing_std = bearing_std
        self.global_localization = global_localization


    def predict(self, u):
        # TODO Implement here the PF, perdiction part
        
        self.particles = np.array([sample_from_odometry(particle, u, self._alphas) for particle in self.particles])
        
        gaussian_parameters = get_gaussian_statistics(self.particles)

        self._state_bar.mu = gaussian_parameters.mu
        self._state_bar.Sigma = gaussian_parameters.Sigma

    def update(self, z):
        # TODO implement correction step
        phi, lm_id = z[0], z[1]

        e_observation = np.array([get_observation(self.particles[i], lm_id)[0] for i in range(self.num_particles)])
        angle_dev = np.array([wrap_angle(phi - e_observation[i]) for i in range(self.num_particles)])
                
        weights = gaussian.pdf(angle_dev, scale=self.bearing_std)
        weights = weights / np.sum(weights)  # normalization

        index = []
        c = weights[0]
        i = 0
        r = uniform(0, 1/self.num_particles)
        for n in range(self.num_particles):
            U = r + n/self.num_particles
            while U > c:
                i += 1
                c += weights[i]
            index += [i]

        self.particles = self.particles[index]
        gaussian_parameters = get_gaussian_statistics(self.particles)

        self._state.mu = gaussian_parameters.mu
        self._state.Sigma = gaussian_parameters.Sigma