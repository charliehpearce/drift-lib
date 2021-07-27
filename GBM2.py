"""
Geometric Brownian Motion 
Ability to vary mu and simga as a function of t
"""

import numpy as np
import math

class Brownian():
    def __init__(self, s0) -> None:
        self.s0 = s0

    def stock_price(
        self,
        mu_fn = lambda x: 1,
        sigma_fn = lambda x: 1,
        deltaT = 52,
        dt = 0.1):
        
        # for n time steps
        n_timesteps = int(deltaT/dt)

        # np array of values
        S = np.zeros(n_timesteps+1)
        # Set first timestep to s0
        S[0] = self.s0

        # Generate sigma and mu vectors
        # range between 0,1 incremented by n_timesteps
        mu_vec = np.array([mu_fn(x/n_timesteps) for x in range(n_timesteps)])
        sigma_vec = np.array([sigma_fn(x/n_timesteps) for x in range(n_timesteps)])

        for t in range(n_timesteps):
            # generate noise 

            # get prev time step (s0 is index 0)
            s = S[t]
            mu = mu_vec[t]
            sig = sigma_vec[t]

            dW = np.random.normal(0,math.sqrt(dt))
            ds = mu*dt*s + sig*s*dW
            S[t+1] = s + ds
        
        # Return vals, not including s0
        return S[1:]

if __name__ == "__main__":
    def mu1(x):
        if x < 230:
            return 0.0001
        #elif 230 <= x < 300:
        #    return  (0.03/70)*(x-230) + 0.22
        else: 
            return 0.0005

    gbm = Brownian(s0=100)
    mu_fn = lambda x: 1
    sigma_fn = lambda x: 1
    pcx = gbm.stock_price(mu_fn=mu1,sigma_fn=sigma_fn, dt=1, deltaT=500)
    print(pcx)