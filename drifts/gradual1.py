class MuSigma:
    """
    Problems:
    Drift parameters are not normalised for the time scale, this means
    it can only be used for tested time frames
    """
    def __init__(self):
        self.meta_data = {
            'drift_type': 'gradual',
            'drift_times': [0.30, 0.65],
            'params_varied': ['mu'],
        }

    def mu(x):
        if x < 0.30:
            return-0.0015
        elif 0.30 <= x <= 0.65:
            # Linear increase
            m = (0.014*(x-0.3))-0.0015
            return m
        else:
            return 0.0020

    def sig(x):
        return 0.0015