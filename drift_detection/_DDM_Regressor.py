"""
Drift Detection Method
Gama et al.
"""
from ._DDM import DDM
import numpy as np

class DDMRegressor(DDM):
    def __init__(self, base_regressor, warning_level=2.0, alarm_level=3.0, min_n_errors=30, min_samples_train=50) -> None:
        super().__init__(warning_level=warning_level, alarm_level=alarm_level, min_n_errors=min_n_errors)
        
        self.base_regressor = base_regressor
        self.min_samples_train = min_samples_train

        self.model_trained = False
        self.window = [[],[]]
    
    def add_element(self, feat, lab):
        self.reset_alarms()
        self.window[0].append(feat)
        self.window[1].append(lab)
        self._apply()

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