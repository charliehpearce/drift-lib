"""
DDM Wrapper for Regressors
"""
from ._base_drift import BaseDrift
import numpy as np

class ECDDResidual(BaseDrift):
    def __init__(self, lmbda=0.2, alarm_level=0.25, min_n_errors=100) -> None:

        super().__init__()
        
        self.lmbda = lmbda
        self.alarm_level = alarm_level
        self.min_n_errors = min_n_errors

        self.mu0 = 0
        self.sig0 = 0
        self.prev_z = 0
        
    def _apply(self):
        if self.n < self.min_n_errors:
            return
        
        # Z_0 = mu_0
        # Collect enough samples to approximate the stream mean
        if self.n == self.min_n_errors:
            self.mu0 = np.mean(self.window)
            self.prev_z = self.mu0
            self.sig0 = np.std(self.window)
            self.window = [] # REFACTOR: implment clear window function in basedrift
            return
        
        z_t = (1-self.lmbda)*self.prev_z + self.lmbda*self.window[-1]
        t = self.n - self.min_n_errors
        
        z_sigma = np.sqrt((self.lmbda/(2-self.lmbda)) * (1 - (1-self.lmbda)**(2*t))*self.sig0)
        
        # Check drift
        if z_t > self.mu0 + self.alarm_level*z_sigma:
            self._drift_alarm = True
            self.reset()
            
        
        # Clear window
        self.window = []

    def reset(self):
        """
        Reset all errors
        """
        self.n = 0
        self.window = []
        self.mu0 = 0
        self.sig0 = 0
        self.prev_z = 0