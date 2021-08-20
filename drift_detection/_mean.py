from ._ADWIN import ADWIN
import numpy as np
from scipy.stats import ttest_ind

class MeanDrift(ADWIN):
    def __init__(self) -> None:
        super().__init__()

    def test_stat(self, window1, window2):
        #e = self.e_cut(len(window1),len(window2))
        return ttest_ind(window1,window2)[1]
