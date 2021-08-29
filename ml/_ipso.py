"""
PSO 
https://www.youtube.com/watch?v=JhgDMAm-imI
Useful article on PSO
https://towardsdatascience.com/particle-swarm-optimization-visually-explained-46289eeb2e14
"""
from abc import abstractmethod
import numpy as np
from numpy.lib.shape_base import tile

"""
To Do:
Refactor 
"""

class PSO:
    def __init__(self, max_epoch=100, c1=2, c2=1.5, w=0.7, verbose=False) -> None:
        self.max_epoch = max_epoch
        self._epoch = 0
        self._c1 = c1
        self._c2 = c2
        self._w = w
        self._solution_found = False
        self._verbose = verbose

    def _update_particles(self):
        # add inertia
        inertia = self._w * self._velocities
        # add cognitive component
        r_1 = np.random.rand(*self.particles.shape)
        cog = self._c1 * r_1 * (self._p_bests - self.particles)
        
        # add social component
        r_2 = np.random.rand(*self.particles.shape)
        #(self._N, *[1 for _ in range(len(self._g_best.shape))]) OLD
        g_best_stack_dim = (self._N, *[1]*len(self._g_best.shape))

        g_best = np.tile(self._g_best[None], g_best_stack_dim)
        glob = self._c2 * r_2 * (g_best  - self.particles)

        new_velocities = inertia + cog + glob
        
        # update positions and velocities
        self.velocities = new_velocities
        self.particles = self.particles + new_velocities

    def _update_best(self):
        """
        Update global and personal bests after moving particles
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
        true_idxs = np.nonzero(t_f_diff)
        self._p_bests[true_idxs] = self.particles[true_idxs]
        
        if self._verbose:
            print(f'Epoch number: {self._epoch} \nGlobal Best fitness: {self._g_best_val} Epoch best fitness:{fitness_scores[g_best_idx]} with std {np.std(fitness_scores)}')

    def _update_coeffs(self):
        """
        Reduces the personal component of the swarm towards
        the end of the swarm, as well as increasing glob component
        and reducing w.
        """
        t = self._epoch/self.max_epoch
        self._c1 = (-3*t) + 3.5
        self._c2 = (3*t) + 0.5
        self._w = (0.4/self.max_epoch**2) * (self._epoch - self.max_epoch) ** 2 + 0.4

    @abstractmethod
    def fitness_fn(self):
        """
        Fitness_fn should accept the particles and return an
        array with the value to be optimised for e.g RMSE.
        """
        raise NotImplementedError

    def optimize(self, particles):
        """
        Particles should be any size numpy mtrx
        """
        # Init variables from  particles
        self.particles = np.array(particles)
        self._velocities = np.random.uniform(-1,1, self.particles.shape)
        self._N = len(self.particles)
        
        self._p_bests = self.particles
        self._p_best_values = np.array(self.fitness_fn(self.particles))
        g_best_idx = np.argmin(self._p_best_values)
        self._g_best_val = self._p_best_values[g_best_idx]
        self._g_best = self.particles[g_best_idx]
        
        ## main loop 
        while (self.max_epoch > self._epoch) and not self._solution_found:
            self._update_particles()
            self._update_best()
            self._update_coeffs()
            self._epoch += 1
        
        #print(self._p_bests)
        return self._g_best

if __name__ == "__main__":
    # This could do with a refactor - do in evening
    class MinFinder(PSO):
        def __init__(self) -> None:
            
            super().__init__(max_epoch=50,verbose=False)
            self.particles = np.random.uniform(-5, 5, (50, 2))

            self.best = self.optimize(self.particles)
            print(self.best)
            

        def fitness_fn(self, particles):
            fitness = []
            for p in particles:
                x = p[0]
                y = p[1]
                fit = x ** 2 + (y + 1) ** 2 - 5 * np.cos(1.5 * x + 1.5) - 5 * np.cos(2 * y - 1.5)
                fitness.append(fit)
            return fitness
    
    fitness_test = MinFinder()
    tst = fitness_test.fitness_fn(particles=[fitness_test.best,[-0.66020389, -1.99281786]])
    print(tst)