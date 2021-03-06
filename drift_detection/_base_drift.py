"""
Drift Super Class
"""
from abc import abstractmethod

class BaseDrift:
    def __init__(self) -> None:
        self._drift_alarm = False
        self._drift_warning = False
        self.window = []
        self.n = 0
    
    @property
    def drift_alarm(self):
        return self._drift_alarm

    @property
    def drift_warning(self):
        return self._drift_warning

    @abstractmethod
    def _apply(self):
        raise NotImplementedError

    def add_element(self, p):
        self.reset_alarms()
        self.n += 1
        self.window.append(p)
        self._apply()
        
    def reset_alarms(self):
        self._drift_warning = False
        self._drift_alarm = False