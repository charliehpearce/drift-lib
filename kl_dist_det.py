import math
import random
import numpy as np
import seaborn as sns

"""
For each window, run a monte carlo simulation under assumptions of mu and sigma
Can test if second window is 95% outside of distrbution.
"""
# Use KL distance (relative entropy) to measure similarity between windows
class BaseDrift:
    def __init__(self) -> None:
        pass

    def add_element(self):
        pass

    def drift_alarm(self):
        # Drift alarm
        pass

    def drift_detected(self):
        # Drift detection
        pass

class KLAdwin(BaseDrift):
    """
    Sliding window. When drift has been detected, data from the previous
    window is removed.

    Ideas/notes:
    Can use the window for a new ml algo.
    Bootstrap data from windows?
    Confidence level alpha from KL distance - VIA BOOTSTRAP
    If KL distance is increasing between windows at a parametric rate then gradual drift may be occuring
    
    Need to bin data somehow so bins are comparible NO
    https://www.youtube.com/watch?v=Kho4VuKmQdE
    Monte carlo simulation under 
    """
    def __init__(self, 
        alpha = 0.05, 
        window_size = 0.002, 
        n_bootstrap = 10000, 
        max_window_size = 500,
        min_window_size = 50,
        n_bins = 10,
        min_subwindow_size = 20,) -> None:
        """
        Parameters
        Alpha: Alpha for generating confidence intervals
        n_bootstrap: Number of bootstrap samples to be drawn from each dist
        max_window_size: Limit the window size
        min_window_size: Ensure window size is small enough for bootstrap (should this be calculated??)
        n_bins: number of bins when generating probability distributions
        min_subwindow_size: min size for ADWIN subwindow (again, should this be calculated??)
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.n_bins = n_bins
        self.delta = window_size    
        self.max_window_size = max_window_size 
        self.min_window_size = min_window_size   
        self.min_subwindow_size = min_subwindow_size    
        
        self.window_size = 0
        self.t_index = 0
        self.window = []

    @staticmethod
    def generate_pdists(n_bins, dist0:list, dist1:list):
        """
        Arguments: 
        bootstraped data from dist 0 and dist 1 : list(s)

        Returns: Two probabilty distrbutions that have the same bins
        for KL distance calculations : list(s)
        """
        def generate_dist(n_bins, bin_width, dist, min_val):
            # For every data point, add to bin
            binned = np.zeros(n_bins)+0.0000001
            for i in dist:
                r = i%bin_width
                bin_index = int((i-r-min_val)/bin_width)-1
                binned[bin_index] += 1
            # Div by total number to get prob dist
            total_n = np.sum(binned)
            prob_dist = binned/total_n
            return prob_dist

        all_vals = dist0 + dist1
        
        # Get max and min from all vals
        max_val = max(all_vals)
        min_val = min(all_vals)

        bin_width = (max_val-min_val)/(n_bins)

        p_dist0 = generate_dist(n_bins=n_bins, bin_width=bin_width, dist=dist0, min_val=min_val)
        p_dist1 = generate_dist(n_bins=n_bins, bin_width=bin_width, dist=dist1, min_val=min_val)

        return p_dist0,p_dist1

    @staticmethod
    def calculate_kl_distance(d1,d2):
        return np.sum(np.multiply(d1, np.log(d1/d2)))
        
    def get_bootstrap_intervals(self, dist0, dist1):
        # Loop for no bootstrap intervals
        kl_distances = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            # Generate bootstrapped data from dists
            k = 2*len(dist0)
            dist0_bootstrap = random.choices(dist0, k=k)
            dist1_bootstrap = random.choices(dist1, k=k)
            p_dist0, p_dist1 = self.generate_pdists(n_bins=self.n_bins, dist0=dist0_bootstrap , dist1=dist1_bootstrap) 

            # Calculate KL distance and append to list
            kl_distances[i] = self.calculate_kl_distance(np.array(p_dist0), np.array(p_dist1))
        
        sns.histplot(kl_distances)
        # Calculate intervals
        lower_interval = np.quantile(kl_distances, q=self.alpha)
        upper_interval = np.quantile(kl_distances, q=(1-self.alpha))
        return lower_interval, upper_interval
    
    def apply(self):
        pass
        # Calcualte number of splits

        # loop over every split in the window

        # check if there's a change in the distrubtion

        # if there is a change, shrink window size to point of drift
        

if __name__ == "__main__":
    pass

"""
Bootstrap data points
Create probability distributions from each distribution
Compute KL distance
Add to list of KL distances and return if significant

Q: Should the data be drawn from a pool or induvidual
"""