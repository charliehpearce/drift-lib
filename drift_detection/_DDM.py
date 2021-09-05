"""
Drift Detection Method
Gama et al.
"""
from _base_drift import BaseDrift
import numpy as np

class DDM(BaseDrift):
    def __init__(self, warning_level=2.0, alarm_level=3.0, min_n_errors = 30) -> None:
        super().__init__()
        self.warning_level = warning_level
        self.alarm_level = alarm_level
        self.min_n_errors = min_n_errors
        
        # Highest
        self.min_std = np.inf
        self.min_error = np.inf
        self.min_std_error = np.inf
        
        self.error_prob = 1
        self.error_std = 0
        self.error_prob_std = 0

        # [DEBUG]
        self.errors = []

    def _apply(self):
        if self.n < self.min_n_errors:
            return False
        # Go over elements in window and compute probs
        for e in self.window:
            self.error_prob += (e-self.error_prob)/self.n
            self.error_std = np.sqrt(self.error_prob*(1-self.error_prob)/self.n)
        
        # [DEBUG]
        self.errors.append(self.error_prob)
        
        # Clear window
        self.window = []
        
        if (self.error_prob+self.error_std) < self.min_std_error:
            self.min_error = self.error_prob
            self.min_std = self.error_std
            self.min_std_error = self.min_error+self.min_std
        
        if self.error_prob+self.error_std > self.min_error + self.alarm_level*self.min_std:
            self._drift_alarm = True
            self.reset()
        elif self.error_prob+self.error_std > self.min_error + self.warning_level*self.min_std:
            self._drift_warning = True
    
    def reset(self):
        self.min_std = np.inf
        self.min_error = np.inf
        self.min_std_error = np.inf
        self.n = 0

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    r1 = np.random.binomial(1,0.3,1000)
    r2 = np.random.binomial(1,0.7,1000)
    r = np.concatenate((r1,r2))
    
    dd = DDM()
    for i,x in enumerate(r):
        dd.add_element(x)
        if dd.drift_alarm:
            print(f'drift alarm {i}')
    
    plt.plot(dd.errors)
    plt.show()
