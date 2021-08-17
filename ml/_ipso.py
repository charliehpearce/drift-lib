"""
PSO 
https://www.youtube.com/watch?v=JhgDMAm-imI
Useful article on PSO
https://towardsdatascience.com/particle-swarm-optimization-visually-explained-46289eeb2e14
"""
import numpy as np
from numpy.lib.index_tricks import fill_diagonal

# stopping criteria, if avg fitness is not better than rest

class PSO:
    def __init__(self, max_epoch, c1=1, c2=2, w=0.78) -> None:
        
        
        self.max_epoch = max_epoch

        self._epoch = 0
        self._c1 = c1
        self._c2 = c2
        self._w = w
        self._N = len(self.particles)
        self._solution_found = False
        self.optimize()

        # what are the initial velocities? Randomise
    def update_particles(self):
        # Inertia Component
        inertia = self._w * self._velocities

        # Personal Compoenent (cog)
        r1 = np.random.random(self._N) 
        cog = (self.c1 * r1) * (self._p_bests - self.particles)
        
        # Global Component
        r2 = np.random.random(self.N)
        g_best_stack_dim = (self.N, *[1 for _ in range(len(self._g_best.shape))])
        tiled_gbest = np.tile(self._g_best, g_best_stack_dim)
        assert tiled_gbest.shape == self.particles.shape
        glob = (self.c2 * r2) * (tiled_gbest- self.particles)

        new_vels = inertia + cog + glob

        # check to see if there has been a change
        if self._velocities - new_vels == 0:
            # No change in velocities suggests solution has been found
            self._solution_found == True
        
        # set vels to new vels
        self.velocities = new_vels
        self.particles += new_vels

    def update_best(self):
        """
        Note: This will need to be updated if maxima is needed
        instead of a minima
        """
        # Call fitness fn
        fitness_scores = self.fitness_fn()
        # Get min value idx (if optimising for minima)
        g_best_idx = np.argmin(fitness_scores)
        self.g_best = self.particles[g_best_idx]

        # update personal bests
        # Get true/false 
        t_f_diff = self._p_best_values < fitness_scores
        # get idx of true false
        true_idxs = np.nonzero(t_f_diff)[0] 
        # Set new best vals
        self._p_bests[true_idxs] = self.particles[true_idxs]

    def optimize(self, particles, fitness_fn):
        """
        Particles should be any size numpy mtrx
        Fitness_fn should accept the particles and return an
        array with the value to be optimised for e.g RMSE.
        """
        # Init variables from  particles
        self.particles = np.array(particles)
        self.fitness_fn = fitness_fn
        self._velocities = np.random.rand(*particles.shape)
        self._p_bests = self.particles
        self._p_best_values = np.array((particles.shape[0],1))
        self._g_best = self.particles[0]
        
        # Update best after init
        self.update_best()

        ## main loop 
        # move particles
        while (self.max_epoch > self._epoch) and not self._solution_found:
            self.update_particles()
            self.update_best()
            self._epoch += 1

        return self.g_best