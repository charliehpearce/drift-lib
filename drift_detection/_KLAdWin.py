from ._ADWIN import ADWIN
import numpy as np
import random

class KLAdWin(ADWIN):
    def __init__(self, n_bins = 20, delta = 1.5) -> None:
        """
        n_bins for creating probabilty distibution functions
        """
        super().__init__(delta=delta)
        self.n_bins = n_bins

    def test_stat(self, window1, window2):
        pdf1, pdf2 = self.create_pdf(window1,window2)
        return self.calc_kl_distance(pdf1,pdf2)
    
    @staticmethod
    def calc_kl_distance(d1,d2):
        return np.sum(np.multiply(d1, np.log(d1/d2)))

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

    def get_bootstrap_intervals(self, dist0, dist1):
        """
        Used to get bootsrap intervals for KL distance
        (DEPRECATED)
        """
        # Loop for no bootstrap intervals
        kl_distances = np.zeros(self.n_bootstrap)
        
        # Calculate KL distnace from samples
        p_dist0_sample, p_dist1_sample = self.create_pdf(n_bins=self.n_bins, dist0=dist0 , dist1=dist1) 
        kl_sample = self.calculate_kl_distance(p_dist0_sample,p_dist1_sample)
        
        # Calculate bootstrap intervals
        for i in range(self.n_bootstrap):
            # Generate bootstrapped data from dists
            k = 2*len(dist0)
            dist0_bootstrap = random.choices(dist0, k=k)
            dist1_bootstrap = random.choices(dist1, k=k)
            p_dist0, p_dist1 = self.generate_pdists(n_bins=self.n_bins, dist0=dist0_bootstrap , dist1=dist1_bootstrap) 

            # Calculate KL distance and append to list
            kl_distances[i] = self.calculate_kl_distance(np.array(p_dist0), np.array(p_dist1))
        
        # Calculate intervals
        lower_interval = np.quantile(kl_distances, q=(self.alpha/2))
        upper_interval = np.quantile(kl_distances, q=(1-(self.alpha)/2))
        mean = np.mean(kl_distances)
        median = np.median(kl_distances)
        
        return {'kl_sample':kl_sample, 'mean':mean, 'median':median,\
             'lower_interval': lower_interval, 'upper_interval': upper_interval}