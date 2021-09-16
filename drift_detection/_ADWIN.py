"""
ADaptive WINdowing super class
Set test_stat method to a function returning a p_val, test stat etc etc
"""

from abc import abstractmethod
from ._base_drift import BaseDrift

class ADWIN(BaseDrift):
    def __init__(self, minimum_window_size=200, minimum_subwindow_size=75,\
         maximum_window_size = 2000, delta=20, persistence_factor=3, step_size = 50) -> None:
        
        super().__init__()

        self.minimum_window_size = minimum_window_size
        self.minimum_subwindow_size = minimum_subwindow_size
        self.delta = delta
        self.persistence_factor = persistence_factor
        self.maximum_window_size = maximum_window_size
        self.step_size = step_size

        self.prev_drift_loc = 0
        self.persist = 0
        

    @abstractmethod
    def test_stat():
        """
        Return test statistic to comapre to delta
        """
        return NotImplementedError

    def _apply(self):
        # Remove 0th element if bigger than max size
        if len(self.window) > self.maximum_window_size:
            self.window.pop(0)
        
        # Persistence factor
        self.reset_alarms()

        if len(self.window) > self.minimum_window_size:
            # Loop over window from start of subwindow, steps can be adjusted 
            for i in range(self.minimum_subwindow_size,
                len(self.window)-self.minimum_subwindow_size,
                self.step_size):
                
                window_0 = self.window[:i]
                window_1 = self.window[i:]

                tstat = self.test_stat(window_0,window_1)

                if tstat <= self.delta:
                    self._drift_warning = True
                    self.persist += 1
                else:
                    self.persist = 0
                
                if self.persist >= self.persistence_factor:
                    self._drift_alarm = True
                    #Print drift location
                    self.prev_drift_loc = self.n-(len(self.window)-i)
                    self.window = window_1
                    break