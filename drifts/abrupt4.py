class MuSigma:
    """
    Problems:
    Drift parameters are not normalised for the time scale, this means
    it can only be used for time frames generated
    """
    def __init__(self):
        self.meta_data = {
            'drift_type': 'abrupt',
            'drift_times': [0.42],
            'params_varied': ['mu','sigma'],
        }

    def mu(x):
        if x < 0.42:
            return 0.0010
        else:
            return -0.0004

    def sig(x):
        if x < 0.42:
            return 0.0010
        else: 
            return 0.0015