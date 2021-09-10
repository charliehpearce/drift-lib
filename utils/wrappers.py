import numpy as np
import sys

class ResidualDriftHelper:
    def __init__(self, regressor, drift_detector, min_samples_train) -> None:
        self.regressor = regressor
        self.drift_detector = drift_detector
        self.min_samples_train = min_samples_train
        
        self.feat_labs = [[],[]]
        self.model_trained = False

    @property
    def drift_alarm(self):
        return self.drift_detector.drift_alarm

    def add_element(self, feat, label):        
        
        if self.model_trained == False:
            # Reset drift alarms 
            self.drift_detector.reset_alarms()
            # Collect data to train regressor
            self.feat_labs[0].append(feat)
            self.feat_labs[1].append(label)

            # Train model if enough data
            if len(self.feat_labs[0]) >= self.min_samples_train:
                # Train
                self.regressor.fit(self.feat_labs[0],self.feat_labs[1])
                self.model_trained = True
                # Clear window
                self.feat_labs = [[],[]]
            else:
                return False
        
        # Model trained
        else:
            y_hat = self.regressor.predict(feat)
            residual = y_hat - label
            self.drift_detector.add_element(residual)
            
            if self.drift_detector.drift_alarm:
                self.model_trained = False

class PIDriftHelper:
    def __init__(self, pi_regressor, drift_detector, min_samples_train) -> None:
        self.regressor = pi_regressor
        self.drift_detector = drift_detector
        self.min_samples_train = min_samples_train
        
        self.feat_labs = [[],[]]
        self.model_trained = False
        self.bound1 = []
        self.bound2 = []

    def get_preds(self):
        return self.regressor.preds

    @property
    def drift_alarm(self):
        return self.drift_detector.drift_alarm

    def add_element(self, feat, label):        
        
        if self.model_trained == False:
            # Reset drift alarms 
            self.drift_detector.reset_alarms()
            # Collect data to train regressor
            self.feat_labs[0].append(feat)
            self.feat_labs[1].append(label)

            # Train model if enough data
            if len(self.feat_labs[0]) >= self.min_samples_train:
                # Train
                self.regressor.fit(np.array(self.feat_labs[0]),np.array(self.feat_labs[1]))
                self.model_trained = True
                # Clear window
                self.feat_labs = [[],[]]
            else:
                return False
        
        # Model trained
        else:
            lower_bound, upper_bound = self.regressor.predict(feat)
            self.bound1.append(lower_bound)
            self.bound2.append(upper_bound)
            
            if lower_bound <= label <= upper_bound:
                self.drift_detector.add_element(0)
            else:
                self.drift_detector.add_element(1)
            
            if self.drift_detector.drift_alarm:
                self.model_trained = False