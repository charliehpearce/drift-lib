class MuSigma:
    """
    Problems:
    Drift parameters are not normalised for the time scale, this means
    it can only be used for time frames generated
    """
    def __init__(self):
        self.meta_data = {
            'drift_type': 'abrupt',
            'drift_times': [0.42,0.55],
            'params_varied': ['sigma','mu'],
        }

    def mu(x):
        if x < 0.42:
            return 0.0010
        elif 0.42 <= x <= 0.55: 
            return 0.0001
        else:
            return 0.0014

    def sig(x):
        if x < 0.42:
            return 0.0015
        elif 0.42 <= x <= 0.55: 
            return 0.0020
        else: 
            return 0.0010