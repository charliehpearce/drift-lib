from ._ADWIN import ADWIN
import numpy as np
import random
from scipy.integrate import simpson
from scipy.stats import gaussian_kde

class KLAdWin(ADWIN):
    def __init__(self, n_bins = 50, delta = 0.1, **kwargs) -> None:
        """
        n_bins for creating probabilty distibution functions
        """
        super().__init__(delta=delta, **kwargs)
        self.n_bins = n_bins

    def test_stat(self, window1, window2):
        pdf1, pdf2, x = self.create_pdf(window1,window2)
        kl = self.calc_kl_distance(pdf1,pdf2,x)
        return kl
    
    @staticmethod
    def calc_kl_distance(pdf1, pdf2, x):
        combined = np.multiply(pdf2, np.log2(pdf2/pdf1))
        integral = simpson(combined, x=x, dx=1)
        return integral

    def create_pdf(self, window1:list, window2:list):
        """
        Returns: Two PDFs from the window with the same format
        for KL distance calculations : list(s)
        """
        # Catch DIV0 errors
        epsilon = 0.00000001

        dist_1_kde = gaussian_kde(window1)
        dist_2_kde = gaussian_kde(window2)

        all_vals = window1 + window2
        min_val = min(all_vals)
        max_val = max(all_vals)

        x_eval = np.linspace(min_val, max_val, self.n_bins)

        # Create Prob Density Fns
        dist_1_pdf = dist_1_kde.evaluate(x_eval)+epsilon
        dist_2_pdf = dist_2_kde.evaluate(x_eval)+epsilon

        return dist_1_pdf, dist_2_pdf, x_eval

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