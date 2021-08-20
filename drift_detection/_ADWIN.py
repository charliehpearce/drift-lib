from abc import abstractmethod
from ._base_drift import BaseDrift
import numpy as np

class ADWIN(BaseDrift):
    def __init__(self, minimum_window_size=200, minimum_subwindow_size=50,\
         maximum_window_size = 3000, alpha=0.005, persistence_factor=2) -> None:
        
        super().__init__()
        self.minimum_window_size = minimum_window_size
        self.minimum_subwindow_size = minimum_subwindow_size
        self.alpha = alpha
        self.persistence_factor = persistence_factor
        self.maximum_window_size = maximum_window_size
        
        self.persist = 0

    @abstractmethod
    def test_stat():
        return NotImplementedError

    def apply(self):
        # Remove 0th element if bigger than max size
        if len(self.window) > self.maximum_window_size:
            self.window.pop(0)
        
        # Persistence factor
        self.reset_alarms()

        window_step_size = 50

        if len(self.window) > self.minimum_window_size:
            # Loop over window from start of subwindow, steps can be adjusted 
            for i in range(self.minimum_subwindow_size,
                len(self.window)-self.minimum_subwindow_size,
                window_step_size):
                
                window_0 = self.window[:i]
                window_1 = self.window[i:]

                pval = self.test_stat(window_0,window_1)

                if pval >= (1-self.alpha):
                    self.drift_warning = True
                    self.persist += 1
                else:
                    self.persist = 0
                
                if self.persist >= self.persistence_factor:
                    self.drift_alarm = True
                    self.window = window_1
                    break