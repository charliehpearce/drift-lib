"""
Drift Super Class
"""

from abc import abstractmethod

class BaseDrift:
    def __init__(self) -> None:
        self._drift_alarm = False
        self._drift_warning = False
        self.window = []
        self.t_index = 0
    
    @property
    def drift_alarm(self):
        return self._drift_alarm

    @property
    def drift_warning(self):
        return self._drift_warning

    @abstractmethod
    def apply(self):
        raise NotImplementedError

    def add_element(self, p):
        self.window.append(p)
        self.apply()
        self.t_index += 1
    
    def reset_alarms(self):
        self._drift_warning = False
        self._drift_alarm = False
