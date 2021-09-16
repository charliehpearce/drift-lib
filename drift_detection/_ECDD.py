"""
EWMA for Concept Drift Detection

https://arxiv.org/abs/1212.6018
"""

from ._base_drift import BaseDrift
import numpy as np

class ECDD(BaseDrift):
    def __init__(self, min_instance=30, lda=0.2):
        """
        lda means lambda, due to lambda function in Python
        it couldnt be set to 'lambda'
        """
        super().__init__()

        self.min_n_errors = min_instance

        self.m_n = 1.0
        self.m_sum = 0.0
        self.m_p = 0.0
        self.m_s = 0.0
        self.z_t = 0.0
        self.lda = lda

    def _apply(self):
        self.m_sum += self.window[-1]
        self.m_p = self.m_sum / self.m_n
        self.m_s = np.sqrt(self.m_p * (1.0 - self.m_p) * self.lda * (1.0 - np.power(1.0 - self.lda, 2.0 * self.m_n)) / (2.0 - self.lda))
        self.m_n += 1

        self.z_t += self.lda * (self.window[-1] - self.z_t)
        L_t = 3.97 - 6.56 * self.m_p + 48.73 * (self.m_p**3) - 330.13 * (self.m_p**5) + 848.18 * (self.m_p**7)

        self.window = []

        if self.m_n < self.min_n_errors:
            return 

        if self.z_t > self.m_p + L_t * self.m_s:
            self._drift_alarm = True
            self.reset()
        elif self.z_t > self.m_p + 0.5 * L_t * self.m_s:
            self._drift_warning = True

    def reset(self):
        self.m_n = 1
        self.m_sum = 0
        self.m_p = 0
        self.m_s = 0
        self.z_t = 0