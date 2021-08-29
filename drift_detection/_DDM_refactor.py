"""
Drift Detection Method
Gama et al.
"""

from ._base_drift import BaseDrift
import numpy as np

class DDM(BaseDrift):
    def __init__(self, warning_level=2.0, alarm_level=3.0, min_n_errors = 30) -> None:
        super().__init__()
        self.warning_level = warning_level
        self.alarm_level = alarm_level
        self.min_n_errors = min_n_errors
        
        self.residual_errors = []
        self.min_std = np.inf
        self.min_error = np.inf
        self.min_std_error = np.inf
        self.N = 1
    
    def _add_error(self, error):
        self.reset_alarms()
        self.residual_errors.append(error)
        
        # Check to see if enough errors
        if self.N > self.min_n_errors:
            self._ddm()

    def _ddm(self, error):
        
        # Calcualte new std
        errors_std = np.std(self.residual_errors)
        
        # Get error from error residuals
        error = self.residual_errors[-1]

        # Update min std and error
        if error+errors_std <= self.min_std_error:
            self.min_std = errors_std
            self.min_error = error
            self.min_std_error = errors_std + error
        
        # Check to see if error occured
        if error + errors_std > self.min_error+self.warning_level*self.min_std:
            self.drift_warning = True
        elif error + errors_std > self.min_error+self.alarm_level*self.min_std:
            self.drift_alarm = True