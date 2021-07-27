class MuSigma:
    """
    Problems:
    Drift parameters are not normalised for the time scale, this means
    it can only be used for tested time frames
    """
    def __init__(self):
        self.meta_data = {
            'drift_type': 'abrupt',
            'drift_times': [0.40, 0.50],
            'params_varied': ['mu','sigma'],
        }

    def mu(x):
        if x < 0.40:
            return-0.0007
        elif 0.40 <= x <= 0.50: 
            return 0.0020
        else:
            return 0.0020

    def sig(x):
        if x < 0.40:
            return 0.0045
        elif 0.40 <= x <= 0.50: 
            return 0.0010
        else: 
            return 0.0020