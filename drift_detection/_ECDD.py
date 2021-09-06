"""
EWMA for Concept Drift Detection

https://arxiv.org/abs/1212.6018
"""

from _base_drift import BaseDrift

class ECDD(BaseDrift):
    def __init__(self, base_ml) -> None:
        super().__init__(base_ml=base_ml)
    
    def apply(self):
        pass
