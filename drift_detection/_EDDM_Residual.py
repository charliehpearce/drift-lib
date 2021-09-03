"""
Early Drift Detection Method
"""
from ._base_drift import BaseDrift
import numpy as np

class EDDMResidual(BaseDrift):
    def __init__(self, base_regressor, warning_level=2.0, alarm_level=3.0,\
         min_n_errors=30, min_samples_train=100) -> None:

        super().__init__()
        self.base_regressor = base_regressor
        self.min_samples_train = min_samples_train

        self.warning_level = warning_level
        self.alarm_level = alarm_level
        self.min_n_errors = min_n_errors
        
        self.residual_errors = []
        self.min_mean_error = np.inf
        self.min_std_error = np.inf
        self.N = 1

        self.model_trained = False
        self.window = [[],[]]
    
    def add_element(self, feat, lab):
        self.reset_alarms()
        self.window[0].append(feat)
        self.window[1].append(lab)
        self._apply()

    def _add_error(self, error):
        self.residual_errors.append(error)
        
        # Check to see if enough errors
        if self.N > self.min_n_errors:
            self._ddm(error)
        
        self.N += 1

    def reset(self):
        """
        Reset all errors
        """
        self.mean_error = np.inf
        self.min_std_error = np.inf
        self.N = 1
        self.residual_errors = []

    def _apply(self):
        """"
        Think about input values and the window
        should append features to [[feats],[labels]
        """
        # Wait until window is of certain length
        if not self.model_trained:
            if (len(self.window[0]) < self.min_samples_train):
                return
            # Fit to model, set model trained to True
            elif len(self.window[0]) == self.min_samples_train:
                # train model
                self.base_regressor.fit(np.array(self.window[0]),np.array(self.window[1]))
                # Clear window and set model trained to true
                self.model_trained = True
        
        else:
            #compute redisudal and add
            y_hat = self.base_regressor.predict(self.window[0][-1])
            residual = y_hat - self.window[1][-1]
            # add residual to DDM
            self._add_error(residual)
            if self._drift_alarm:
                self.model_trained = False
                self.reset()
            
            # Clear window after every iteration, not v efficent but rolling with for now
            self.window = [[],[]]


    def _ddm(self, error):
        # Calcualte new mean error and std
        errors_std = np.std(self.residual_errors)
        # Calculate new mean error (should this be done??)
        mean_error = np.mean(self.residual_errors)
        # Get error from error residuals
        error = self.residual_errors[-1]
        
        # Update min std and mean error (closest to zero)
        if abs(mean_error) <= abs(self.min_mean_error):
            self.min_mean_error = mean_error
        if errors_std <= self.min_std_error:
            self.min_std_error = errors_std
        
        
        # Check to see if error occured on both tails (elif used instead of or for efficency)
        if error + errors_std > self.min_mean_error+(self.alarm_level*self.min_std_error):
            self._drift_alarm = True
        elif error - errors_std < self.min_mean_error-(self.alarm_level*self.min_std_error):
            self._drift_alarm = True
        
        # If a drift alarm hasnt been raised, check for warnings
        elif error + errors_std > self.min_mean_error+(self.warning_level*self.min_std_error):
            self._drift_warning = True
        elif error - errors_std < self.min_mean_error-(self.warning_level*self.min_std_error):
            self._drift_warning = True
        