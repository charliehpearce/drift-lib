"""
DDM Wrapper for Regressors
"""
from ._base_drift import BaseDrift
import numpy as np

class DDMResidual(BaseDrift):
    def __init__(self, min_n_errors=30, lmbda=0.1) -> None:

        super().__init__()

        self.lmbda = lmbda
        self.min_n_errors = min_n_errors
        
        self.min_mean_error = np.inf
        self.min_std_error = np.inf

    def _apply(self):
        if self.n < self.min_n_errors:
            return
        
        self.mean_error = np.mean(self.window)
        
        # Calcualte new mean error and std
        errors_std = np.std(self.window)
        # Calculate new mean error 
        mean_error = np.mean(self.window)
        # Get error from error residuals
        error = self.window[-1]
        
    
    def reset(self):
        """
        Reset all errors
        """
        self.mean_error = np.inf
        self.min_std_error = np.inf
        self.n = 0
        self.window = []