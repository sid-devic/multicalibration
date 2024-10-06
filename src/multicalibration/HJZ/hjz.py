import numpy as np
from tqdm import trange
from multicalibration.HJZ.online_learners import Hedge, MLProd, OnlineGradientDescent, OptimisticHedge
from multicalibration.HJZ.adversary import update_hedge_algorithms, Adversary
from multicalibration.HJZ.utils import discretize_values, calculate_average_labels

ALG_CLASSES = {
    'Hedge': Hedge,
    'MLProd': MLProd,
    'OnlineGradientDescent': OnlineGradientDescent,
    'OptimisticHedge': OptimisticHedge,
    'None': None,
}


class HJZAlgorithm:
    """
    General Multicalibration Predictor class.
    Based on the HJZ implementation by Eric Zhao: 
    https://github.com/ericzhao28/multicalibration?tab=readme-ov-file    
    """
    
    def __init__(self):
        """
        Initialize Multicalibration Predictor.
        """        
        # constants in original implementations
        self.base_lr = 1
        self.base_other_lr = 100

    def fit(self, confs, labels, subgroups, params):
        """
        HJZ implementation updates training / validation 
        """
        try:
            # parameters
            self.alg_class = ALG_CLASSES[params['algorithm']]
            self.other_alg_class = ALG_CLASSES[params['other_algorithm']]
            self.lr, self.other_lr = params['lr'], params['other_lr']
            self.n_bins = params['n_bins']
            self.iterations = params['iterations']
        except KeyError as e:
            raise ValueError(f"Missing parameter: {e}. Please provide all required parameters (algorithm, other_algorithm, lr, other_lr, n_bins, iterations) as a dictionary.")

        adv = None
        self.learning_rate = lambda x: self.base_lr * np.power(self.lr, x)
        self.other_learning_rate = lambda x: self.base_other_lr * np.power(self.other_lr, x)
        groups_train = subgroups

        # 1-hot encode labels
        c = self._populate_probs(confs.copy())
        l = self._populate_probs(labels.copy())
        self.target_dim = len(l[0])

        # track for prediction
        self.bad_bins = []
        self.bad_groups = []
        self.bad_classes = []
        self.underestimates = []

        # initialize predictions for each sample, using f0=confs
        # FIXME: potential issue in how we initialize confidences
        train_algs = [
            self.alg_class(len(l[0]), 
                           learning_rate=self.learning_rate(1), 
                           f0=f) for f in c # initialize with confidences
        ]

        for i in trange(1, self.iterations + 1):
            predictions = np.array([alg.weights for alg in train_algs])
            bins = discretize_values(predictions, self.n_bins)
            avg_train_y, counts = calculate_average_labels(
                l, bins, groups_train, self.n_bins
            )
            if adv is None:
                adv = Adversary(
                    self.other_alg_class,
                    avg_train_y.flatten().shape[0] * 2,
                    self.other_learning_rate(i),
                )

            (
                bad_bin,
                bad_group,
                bad_class,
                underestimate,
            ), train_error = adv.find_bad_bin_group(
                avg_train_y, predictions, counts, bins, groups_train, self.other_learning_rate(i)
            )
            
            self.bad_bins.append(bad_bin)
            self.bad_groups.append(bad_group)
            self.bad_classes.append(bad_class)
            self.underestimates.append(underestimate)

            _ = update_hedge_algorithms(
                train_algs,
                bad_bin,
                bad_group,
                bad_class,
                underestimate,
                bins,
                groups_train,
                self.learning_rate(i),
            )

    def predict(self, f_xs, groups):
        """
        Returns calibrated predictions for a batch of data points.
        """
        groups_eval = groups
        c = self._populate_probs(f_xs.copy())
        algs = [
            self.alg_class(self.target_dim, 
                           learning_rate=self.learning_rate(1), 
                           f0=f) for f in c
        ]

        # iterative updates
        for i in trange(1, self.iterations + 1):
            # get bins
            preds = np.array([alg.weights for alg in algs])
            bins = discretize_values(preds, self.n_bins)
            
            (
                bad_bin, bad_group, bad_class, underestimate
            ) = (
                self.bad_bins[i-1],
                self.bad_groups[i-1],
                self.bad_classes[i-1],
                self.underestimates[i-1],
            )

            _ = update_hedge_algorithms(
                        algs,
                        bad_bin,
                        bad_group,
                        bad_class,
                        underestimate,
                        bins,
                        groups_eval,
                        self.learning_rate(i),
                    )

        return np.array([alg.weights for alg in algs])[:, 1]
    

    def _populate_probs(self, probs):
        """
        Populate probabilities across classes, given positive-class confidences.
        """
        return np.array([1 - probs, probs]).T