import math
import random
import numpy as np
import seaborn as sns

"""
For each window, run a monte carlo simulation under assumptions of mu and sigma
Can test if second window is 95% outside of distrbution.
"""

class KLAdwin:
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
        n_bootstrap = 1000, 
        n_bins = 10,
        max_window_size = 500,
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
        self.max_window_size = max_window_size
        self.min_subwindow_size = min_subwindow_size    
        
        self.window_size = 0
        self.t_index = 0
        self.window = []
        self.drift_alarm = False
        self.drift_detected = False
        self.persistence_factor = 5

    @staticmethod
    def generate_pdists(n_bins, dist0:list, dist1:list):
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

        all_vals = dist0 + dist1
        
        # Get max and min from all vals
        max_val = max(all_vals)
        min_val = min(all_vals)

        bin_width = (max_val-min_val)/(n_bins)

        p_dist0 = generate_dist(n_bins=n_bins, bin_width=bin_width, dist=dist0, min_val=min_val)
        p_dist1 = generate_dist(n_bins=n_bins, bin_width=bin_width, dist=dist1, min_val=min_val)

        return p_dist0,p_dist1

    @staticmethod
    # Kullback-Leibler Divergence 
    def calculate_kl_distance(d1,d2):
        return np.sum(np.multiply(d1, np.log(d1/d2)))
        
    def get_bootstrap_intervals(self, dist0, dist1):
        # Loop for no bootstrap intervals
        kl_distances = np.zeros(self.n_bootstrap)
        
        # Calculate KL distnace from samples
        p_dist0_sample, p_dist1_sample = self.generate_pdists(n_bins=self.n_bins, dist0=dist0 , dist1=dist1) 
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
        
        sns.histplot(kl_distances)
        # Calculate intervals
        lower_interval = np.quantile(kl_distances, q=(self.alpha/2))
        upper_interval = np.quantile(kl_distances, q=(1-(self.alpha)/2))
        mean = np.mean(kl_distances)
        median = np.median(kl_distances)
        
        return {'kl_sample':kl_sample, 'mean':mean, 'median':median,\
             'lower_interval': lower_interval, 'upper_interval': upper_interval}

    def add_element(self, point:float):
        self.window.append(point)

        # Remove 0th element if bigger than window size
        if len(self.window) > self.max_window_size:
            self.window.pop(0)

        self.t_index += 1

        # Call adwin, should this be done every nth elements to reduce 
        # computational cost?
        if len(self.window) > 2 * self.min_subwindow_size:
            self.adwin()

    def drift_alarm(self):
        return self.drift_alarm

    def drift_detected(self):
        return self.drift_detected

    # Adwin stuff
    def adwin(self):
        # Every time an element is added, loop over windows.

        # Calculate number of subwindows
        no_iters = len(self.window) - 2*self.min_subwindow_size
        persist_count = 0

        for i in range(no_iters):
            # Split into two dists
            dist0 = self.window[:self.min_subwindow_size + i]
            dist1 = self.window[self.min_subwindow_size + i:]

            result = self.get_bootstrap_intervals(dist0,dist1)

            # Alarm drift if above KL threshold
            if result['kl_sample'] >= 0.4:
                persist_count += 1
                self.drift_alarm = True
            else:
                persist_count = 0

            # If drift detected for more than n persistance factors
            # trigger detected and set the window equal to the 2nd dist
            if persist_count >= self.persistence_factor:
                self.drift_detected = True
                self.window = dist1
                break
        

if __name__ == "__main__":
    pass

"""
Bootstrap data points
Create probability distributions from each distribution
Compute KL distance
Add to list of KL distances and return if significant

Q: Should the data be drawn from a pool or induvidual
"""