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
        
        self.min_std = np.inf
        self.min_error = np.inf
        self.min_std_error = np.inf

        # Used to track how many errors there are
        self.n_errors = 1
        
        self.error_prob = 0
        self.error_prob_std = 0
    
    def _add_error(self, error):
        pass

    def add_element(self, p):
        self._add_error(p)

    def _ddm(self, error):
        pass
    
    def reset(self):
        self.min_std = np.inf
        self.min_error = np.inf
        self.min_std_error = np.inf
        self.n_errors = 1
        self.residual_errors = []