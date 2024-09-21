import numpy as np
from tqdm import trange
from multicalibration.HKRR.hkrr import HKRRAlgorithm
from multicalibration.HJZ.hjz import HJZAlgorithm


class MulticalibrationPredictor:
    """
    General Multicalibration Predictor class.
    """
    
    def __init__(self, algorithm, params=None):
        """
        Initialize Multicalibration Predictor.
        """
        self.algorithm = algorithm
        self.params = params
        if algorithm == 'HKRR':
            self.mcbp = HKRRAlgorithm(params)
        elif algorithm == 'HJZ':
            self.mcbp = HJZAlgorithm(params)
        else:
            raise ValueError(f"Algorithm {algorithm} not supported")

    def fit(self, confs, labels, subgroups):
        """
        Returns vector of confidences on calibration set.

        HKRR: alpha, lmbda, use_oracle=True, randomized=True, max_iter=float('inf')
        """
        # Check if labels are binary
        if len(np.unique(labels)) > 2:
            raise ValueError("Labels must be binary. Multiclass not supported (yet).")
        
        self.mcbp.fit(confs, labels, subgroups)

    def batch_predict(self, f_xs, groups):
        """
        Returns calibrated predictions for a batch of data points.
        HKRR: early_stop=None
        """
        return self.mcbp.batch_predict(f_xs, groups)