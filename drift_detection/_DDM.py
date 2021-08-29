"""
Drift Detection Method
"""

from numpy.core.fromnumeric import shape
from ._base_drift import BaseDrift
import numpy as np

class DDM(BaseDrift):
    def __init__(self, base_ml) -> None:
        super().__init__()
        self.model_trained = False
        self.model_train_size = 100
        self.base_ml = base_ml

        self.errors = []
        
        self.mse_min = np.inf
        self.std_min = np.inf
        self.mse_std_min = np.inf

        # Used to store features and labels for training
        self.feat_window = []
        self.lab_window = []

        self.globerrs = []

        # Predicted 
        self.y_hat = 0
        self.N = 1

    def _train_model(self):
        self.base_ml.fit(np.array(self.feat_window),np.array(self.lab_window))
        self.model_trained = True
        self.y_hat = self.base_ml.predict(self.feat_window[-1])
        print('model training')
        self.reset()

    
    def add_element(self, features, label):
        self.reset_alarms()
        # If model trained == false, check to see if enough data 
        # in window to train model. If not: pass
        if self.model_trained == False:
            if len(self.feat_window) >= self.model_train_size:
                self._train_model()
            else: 
                # Not enough data to train, add to window
                self.feat_window.append(features)
                self.lab_window.append(label)
        else:
            self._ddm(features=features, label=label)
    
    def _compute_mse(self, error):
        """
        Standard dev of errors,
        Error should be y_hat - y
        """
        self.errors.append(error)
        err_arr = np.array(self.errors)
        mse = np.sum(err_arr)/self.N
        std = np.std(err_arr)
        return mse, std

    def _ddm(self, features, label):
        error = self.y_hat - label
        self.y_hat = self.base_ml.predict(features)
        mse, std = self._compute_mse(error)
        self.globerrs.append(error)

        if self.N > 0:
            #check to see if mse or std is lowest
            if (mse + std) < self.mse_std_min:
                self.mse_min = mse
                self.std_min = std
                self.mse_std_min = mse + std
            
            
            # See if drift has occured, set alarms of super class if true
            if mse + std > (self.mse_min + 3*self.std_min):
                # Setting this will start training the model again
                self.model_trained = False
                self._drift_alarm = True
                
            elif mse + std > self.mse_min + 2*self.std_min:
                self._drift_warning = True
        
        self.N += 1
    
    def reset(self):
        self._drift_alarm = False
        self._drift_warning = False
        self.errors = []
        self.window = []
        self.feat_window = []
        self.lab_window = []

        self.mse_min = np.inf
        self.std_min = np.inf
        self.mse_std_min = np.inf
        self.N = 1



"""
Add element should take in an array of [[train],[label]] (maybe add assert statement somewhere?)
"""