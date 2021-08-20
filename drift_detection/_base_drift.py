from abc import abstractmethod


class BaseDrift:
    def __init__(self) -> None:
        self.drift_alarm = False
        self.drift_warning = False
        self.window = []
    
    def drift_alarm(self):
        return self.drift_alarm

    def drift_warning(self):
        return self.drift_warning

    @abstractmethod
    def apply(self):
        raise NotImplementedError

    def add_element(self, p):
        self.window.append(p)
        self.apply()
    
    def reset_alarms(self):
        self.drift_warning = False
        self.drift_alarm = False
