"""
Early Drift Detection Method
Gama et al.
"""
from ._base_drift import BaseDrift
import numpy as np

class EDDM(BaseDrift):
    def __init__(self, warning_level=0.9, alarm_level=0.8, min_n_errors = 100, debug = True) -> None:
        super().__init__()
        self.warning_level = warning_level
        self.alarm_level = alarm_level
        self.min_n_errors = min_n_errors
        self.debug = debug
        
        # Highest
        self.m2std_max = -np.inf
        
        self.mean_d = 0
        self.std_d = 0
        self.last_e = 0
        self.std_temp = 0
        self.n_errors = 0

        # [DEBUG]
        self.errors = []

    def _apply(self):
        """
        Looks at the distance between two errors (i.e 1s)
        """
        if self.n < self.min_n_errors:
            return 
        # Go over elements in window and compute probs
        
        for e in self.window:
            if e == 1:
                # get distance between errors
                distance = self.n - self.last_e
                self.last_e = self.n
                self.n_errors += 1
                old_mean_d = self.mean_d

                # calcaulte mean error distance
                self.mean_d += (distance-self.mean_d)/self.n_errors

                # Calculate mean error distance std
                self.std_temp += (distance-self.mean_d)*(distance*old_mean_d)
                self.std_d = np.sqrt(self.std_temp/self.n_errors)
                self.m2std = self.mean_d + (2*self.std_d)
        
        # [DEBUG]
        if self.debug:
            self.errors.append(self.mean_d)
        
        # Clear window
        self.window = []
        # Check p max and s max
        if self.m2std > self.m2std_max:
            self.m2std_max = self.m2std

        if (self.m2std/self.m2std_max) < self.alarm_level:
            self._drift_alarm = True
            self.reset()

        elif (self.m2std/self.m2std_max) < self.alarm_level:
            self._drift_warning = True

    def reset(self):
        self.m2std_max = -np.inf
        self.mean_d = 0
        self.std_d = 0
        self.last_e = 0
        self.std_temp = 0
        self.n_errors = 0
        self.n = 0

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    r1 = np.random.binomial(1,0.3,1000)
    r2 = np.random.binomial(1,0.7,1000)
    r = np.concatenate((r1,r2))
    
    dd = EDDM()
    for i,x in enumerate(r):
        dd.add_element(x)
        if dd.drift_alarm:
            print(f'drift alarm {i}')
    
    plt.plot(dd.errors)
    plt.show()


"""
– (p′i + 2 · s′i)/(p′max + 2 · s′max) < α for the warning level. 
Beyond this level, the examples are stored in advance of a possible 
change of context.
– (p′i + 2 · s′i)/(p′max + 2 · s′max) < β for the drift level. Beyond
    this level the concept drift is supposed to be true, the model 
    induced by the learning method is reset and a new model is learnt
    using the examples stored since the warning level triggered. 
    The values for p′max and s′max are reset too.
"""