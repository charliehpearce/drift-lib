from ._ADWIN import ADWIN
import numpy as np
import random

class JSAdWin(ADWIN):
    def __init__(self, n_bins = 30, delta = 0.1) -> None:
        """
        n_bins for creating probabilty distibution functions
        """
        super().__init__(delta=delta, minimum_window_size=300)
        self.n_bins = n_bins

    def test_stat(self, window1, window2):
        pdf1, pdf2 = self.create_pdf(window1,window2)
        return self.calc_jenson_shannon(pdf1,pdf2)
    
    @staticmethod
    def calc_kl_distance(d1,d2):
        return np.dot(d1, np.log2(d1/d2))

    @classmethod
    def calc_jenson_shannon(cls,d1,d2):
        m = 0.5 * (d1 + d2)
        js = 0.5*cls.calc_kl_distance(d1,m) + 0.5 * cls.calc_kl_distance(d2,m)
        return js

    def create_pdf(self, window1:list, window2:list):
        """
        Arguments: 
        bootstraped data from dist 0 and dist 1 : list(s)

        Returns: Two probabilty distrbutions that have the same bins
        for KL distance calculations : list(s)
        """
        def generate_dist(n_bins, bin_width, dist, min_val):
            # For every data point, to avoid div0 add 0.000000001
            binned = np.zeros(n_bins)+0.00000000001
            for i in dist:
                r = i%bin_width
                bin_index = int((i-r-min_val)/bin_width)-1
                binned[bin_index] += 1
            # Div by total number to get prob dist
            total_n = np.sum(binned)
            prob_dist = binned/total_n
            return prob_dist

        all_vals = window1 + window2

        # Get max and min from all vals
        max_val = max(all_vals)
        min_val = min(all_vals)

        bin_width = (max_val-min_val)/(self.n_bins)

        p_dist0 = generate_dist(n_bins=self.n_bins, bin_width=bin_width, dist=window1, min_val=min_val)
        p_dist1 = generate_dist(n_bins=self.n_bins, bin_width=bin_width, dist=window2, min_val=min_val)

        return p_dist0,p_dist1