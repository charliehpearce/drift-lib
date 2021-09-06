"""
DDM Wrapper for Regressors
"""
from ._base_drift import BaseDrift
import numpy as np
import sys

class DDMResidual(BaseDrift):
    def __init__(self, warning_level=2.0, alarm_level=3.0, min_n_errors=30) -> None:

        super().__init__()

        self.warning_level = warning_level
        self.alarm_level = alarm_level
        self.min_n_errors = min_n_errors
        
        self.min_mean_error = np.inf
        self.min_std_error = np.inf

    def _apply(self):
        if self.n < self.min_n_errors:
            return
        
        # Calcualte new mean error and std
        errors_std = np.std(self.window)
        # Calculate new mean error 
        mean_error = np.mean(self.window)
        # Get error from error residuals
        error = self.window[-1]
        
        # Update min std and mean error (closest to zero)
        if abs(mean_error) <= abs(self.min_mean_error):
            self.min_mean_error = mean_error
        if errors_std <= self.min_std_error:
            self.min_std_error = errors_std
        
        # Check to see if error occured on either tail
        if (error + errors_std > self.min_mean_error+(self.alarm_level*self.min_std_error)) or \
            (error - errors_std < self.min_mean_error-(self.alarm_level*self.min_std_error)):
            self._drift_alarm = True
            self.reset()

        # If a drift alarm hasnt been raised, check for warnings
        elif (error + errors_std > self.min_mean_error+(self.warning_level*self.min_std_error)) or \
            (error - errors_std < self.min_mean_error-(self.warning_level*self.min_std_error)):
            self._drift_warning = True
    
    def reset(self):
        """
        Reset all errors
        """
        self.mean_error = np.inf
        self.min_std_error = np.inf
        self.n = 0
        self.window = []