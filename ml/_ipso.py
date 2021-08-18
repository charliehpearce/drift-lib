"""
PSO 
https://www.youtube.com/watch?v=JhgDMAm-imI
Useful article on PSO
https://towardsdatascience.com/particle-swarm-optimization-visually-explained-46289eeb2e14
"""
from abc import abstractmethod
import numpy as np

# stopping criteria, if avg fitness is not better than rest

class PSO:
    def __init__(self, max_epoch, c1=2.05, c2=2, w=0.73) -> None:
        self.max_epoch = max_epoch
        self._epoch = 0
        self._c1 = c1
        self._c2 = c2
        self._w = w
        self._solution_found = False

    def update_particles(self):
        # Inertia Component
        inertia = self._w * self._velocities

        # Personal Compoenent (cog)
        r1 = np.random.rand() #*self.particles.shape
        cog = (self._c1 * r1) * (self._p_bests - self.particles)
        
        # Global Component
        r2 = np.random.rand() #*self.particles.shape
        g_best_stack_dim = (self._N, *[1 for _ in range(len(self._g_best.shape))])
        tiled_gbest = np.tile(self._g_best, g_best_stack_dim)
        glob = (self._c2 * r2) * (tiled_gbest-self.particles)

        new_vels = inertia + cog + glob

        # check to see if there has been a change

        # set vels to new vels
        self._velocities = new_vels
        self.particles += new_vels

    def update_best(self):
        """
        Note: This will need to be updated if maxima is needed
        instead of a minima
        """
        # Call fitness fn
        fitness_scores = self.fitness_fn(self.particles)
        # Get min value idx (if optimising for minima)
        g_best_idx = np.argmin(fitness_scores)

        # check to see if fitness score better than prev glob best
        if fitness_scores[g_best_idx] < self._g_best_val:
            self._g_best = self.particles[g_best_idx]
            self._g_best_val = fitness_scores[g_best_idx]

        # update personal bests
        # Get true/false 
        t_f_diff = self._p_best_values > fitness_scores
        # get idx of true false
        true_idxs = np.nonzero(t_f_diff)[0]
        # Set new best vals
        self._p_bests[true_idxs] = self.particles[true_idxs]

    def _update_coeffs(self):
        t = self._epoch/self.max_epoch
        self._c1 = (-3*t) + 3.5
        self._c2 = (3*t) + 0.5
        self._w = (0.4/self.max_epoch**2) * (self._epoch - self.max_epoch) ** 2 + 0.4

    @abstractmethod
    def fitness_fn(self):
        raise NotImplementedError

    def optimize(self, particles):
        """
        Particles should be any size numpy mtrx
        Fitness_fn should accept the particles and return an
        array with the value to be optimised for e.g RMSE.
        """
        # Init variables from  particles
        self.particles = np.array(particles)
        self._velocities = np.random.rand(*particles.shape)
        self._p_bests = self.particles
        self._p_best_values = np.array(self.fitness_fn(self.particles))
        self._g_best_val = np.min(self._p_best_values)
        self._g_best = self.particles[0]
        self._N = len(self.particles)

        ## main loop 
        while (self.max_epoch > self._epoch) and not self._solution_found:
            print(f'Epoch number {self._epoch}')
            self.update_particles()
            self.update_best()
            self._update_coeffs()
            #self._update_coeffs
            self._epoch += 1

        return self._g_best