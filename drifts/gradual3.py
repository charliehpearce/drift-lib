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
            'params_varied': ['mu','sigma'],
        }

    def mu(t):
        x1 = -0.0015
        x2 = 0.0020
        t1 = 0.3
        t2 = 0.65

        if t < t1:
            return x1
        elif t1 <= t <= t2:
            # Linear increase
            m = (((x2-x1)/(t2-t1))*(t-t1))+x1
            return m
        else:
            return x2

    def sig(t):
        x1 = 0.0015
        x2 = 0.0040
        t1 = 0.3
        t2 = 0.65

        if t < t1:
            return x1
        elif t1 <= t <= t2:
            # Linear increase
            m = (((x2-x1)/(t2-t1))*(t-t1))+x1
            return m
        else:
            return x2