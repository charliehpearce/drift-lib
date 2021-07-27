class MuSigma:
    """
    Problems:
    Drift parameters are not normalised for the time scale, this means
    it can only be used for tested time frames
    """
    def __init__(self):
        self.meta_data = {
            'drift_type': 'abrupt',
            'drift_times': [0.26, 0.55],
            'params_varied': ['mu','sigma'],
        }

    def mu(x):
        if x < 0.26:
            return-0.0007
        elif 0.26 <= x <= 0.55: 
            return 0.0020
        else:
            return -0.0020

    def sig(x):
        if x < 0.26:
            return 0.0015
        elif 0.26 <= x <= 0.55: 
            return 0.0035
        else: 
            return 0.0045