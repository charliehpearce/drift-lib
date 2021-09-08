from scipy.stats.stats import ks_2samp
from ._ADWIN import ADWIN
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid
from scipy.stats import ks_2samp

class KSAdWin(ADWIN):
    def __init__(self, n_bins = 50, delta = 0.1) -> None:
        """
        n_bins for creating probabilty distibution functions
        """
        super().__init__(delta=delta, minimum_window_size=300)
        self.n_bins = n_bins
    
    def test_stat(self, window1, window2):
        p = ks_2samp(window1,window2)[1]
        return p

"""
    @staticmethod
    def calc_ks(cdf1, cdf2, x):
        abs_diff = np.absolute(cdf1-cdf2)
        max_val = np.max(abs_diff)
        return max_val

    @staticmethod
    def calc_cdf(dist, x):
        return cumulative_trapezoid(dist, x=x, dx =1)

    def create_pdf(self, window1:list, window2:list):
        
        Returns: Two PDFs from the window with the same format
        for KL distance calculations : list(s)
        
        dist_1_kde = gaussian_kde(window1)
        dist_2_kde = gaussian_kde(window2)

        all_vals = window1 + window2
        min_val = min(all_vals)
        max_val = max(all_vals)

        x_eval = np.linspace(min_val, max_val, self.n_bins)

        # Create Prob Density Fns
        dist_1_pdf = dist_1_kde.pdf(x_eval)
        dist_2_pdf = dist_2_kde.pdf(x_eval)

        
        # Integral to get Cum Dist Fns
        dist_1_p_dist_fn = 
        dist_2_p_dist_fn = cumulative_trapezoid(dist_2_p_den_fn, x=x_eval, dx =1)
        
        return dist_1_pdf, dist_2_pdf, x_eval
"""