import numpy as np
from tqdm import trange
from multicalibration.HKRR.hkrr import HKRRAlgorithm
from multicalibration.HKRR.hkrr_multiclass import MulticlassHKRRAlgorithm
from multicalibration.HJZ.hjz import HJZAlgorithm


class MulticalibrationPredictor:
    """
    General Multicalibration Predictor class.
    """
    def __init__(self, algorithm, num_classes=2, verbose=False):
        """
        Initialize Multicalibration Predictor.
        """
        self.algorithm = algorithm
        self.num_classes = num_classes
        if self.num_classes < 2:
            raise ValueError("Number of classes must be at least 2.")
        
        if self.algorithm == 'HKRR' and self.num_classes == 2:
            self.mcbp = HKRRAlgorithm(verbose=verbose)
        elif self.algorithm == 'HKRR' and self.num_classes > 2:
            self.mcbp = MulticlassHKRRAlgorithm(n_classes=self.num_classes, verbose=verbose)
        elif self.algorithm == 'HJZ' and self.num_classes == 2:
            self.mcbp = HJZAlgorithm()
        elif self.algorithm == 'HJZ' and self.num_classes > 2:
            raise ValueError("HJZ algorithm only supports binary classification problems.")
        else:
            raise ValueError(f"Multicalibration algorithm {algorithm} not recognized / supported.")

    def fit(self, confs, labels, subgroups, params):
        """
        Returns vector of confidences on calibration set.
        
        @param confs: initial confidences on each class
        @param labels: true labels
        @param subgroups: list of subgroups, each containing indices of samples
        @param params: dictionary of hyperparameters

        HKRR params: alpha, lmbda, use_oracle=True, randomized=True, max_iter=float('inf')
        """
        if len(confs) != len(labels):
            raise ValueError("Number of confidence scores and labels must match.")
    
        if self.num_classes == 2 and not np.all(np.isin(labels, [0, 1])):
            raise ValueError("Labels must be in the set \{ 0, 1 } for binary problems.")
        
        if self.num_classes > 2 and not np.all(np.isin(labels, np.arange(self.num_classes))):
            raise ValueError("Labels must be in the set \{ 0, 1, ..., num_classes-1 } for multiclass problems.")
        
        n = len(confs)
        if self.num_classes > 2 and confs.shape != (n, self.num_classes):
            raise ValueError(f"Confidence scores must have shape (n_samples, {self.num_classes}).")
        
        if not all(np.all(np.isin(sg, np.arange(n))) for sg in subgroups):
            raise ValueError("All subgroups must contain indices only in [0, num_samples-1].")

        self.mcbp.fit(confs, labels, subgroups, params)

    def predict(self, f_xs, groups):
        """
        Returns calibrated predictions for a batch of data points.
        HKRR: early_stop=None
        """
        return self.mcbp.predict(f_xs, groups)