from ._ADWIN import ADWIN
from scipy.stats import ttest_ind

class TTestAdWin(ADWIN):
    def __init__(self, p_val) -> None:
        super().__init__(delta=p_val)

    def test_stat(self, window1, window2):
        return ttest_ind(window1,window2, equal_var=False)[1]
