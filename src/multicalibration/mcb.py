import numpy as np
from tqdm import trange
from multicalibration.HKRR.hkrr import HKRRAlgorithm
from multicalibration.HJZ.hjz import HJZAlgorithm


class MulticalibrationPredictor:
    """
    General Multicalibration Predictor class.
    """
    def __init__(self, algorithm, verbose=False):
        """
        Initialize Multicalibration Predictor.
        """
        self.algorithm = algorithm
        if algorithm == 'HKRR':
            self.mcbp = HKRRAlgorithm(verbose=verbose)
        elif algorithm == 'HJZ':
            self.mcbp = HJZAlgorithm()
        else:
            raise ValueError(f"Multicalibration algorithm {algorithm} not recognized / supported.")

    def fit(self, confs, labels, subgroups, params):
        """
        Returns vector of confidences on calibration set.

        HKRR: alpha, lmbda, use_oracle=True, randomized=True, max_iter=float('inf')
        """
        # Check if labels are binary
        if len(np.unique(labels)) > 2:
            raise ValueError("Labels must be binary. Multiclass not supported (yet).")
        self.mcbp.fit(confs, labels, subgroups, params)

    def predict(self, f_xs, groups):
        """
        Returns calibrated predictions for a batch of data points.
        HKRR: early_stop=None
        """
        return self.mcbp.predict(f_xs, groups)