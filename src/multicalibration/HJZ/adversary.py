"""Multicalibration algorithm for multiclass classification."""
import numpy as np
from multicalibration.HJZ.utils import calculate_calibration_errors


def update_hedge_algorithms(
    hedge_algorithms, bad_bin, bad_group, bad_class, underestimate, bins, groups, lr
):
    """
    Updates the hedge algorithms by increasing the weight of the bad class in the bad group in the bad bin.
    Args:
        hedge_algorithms: list of Hedge objects
        bad_bin: bin number
        bad_group: group number
        bad_class: class number
        underestimate: boolean indicating whether the class is underestimated or overestimated
        bins: (n_algorithms) array of bin numbers for each algorithm
        groups: (n_groups) list of lists of algorithm indices for each group
    """
    updated = []

    # print("LEN ALGS", len(hedge_algorithms))
    # print("LEN BINS", len(bins))

    for i, alg in enumerate(hedge_algorithms):
        group_indicator = 1 if i in groups[bad_group] else 0
        bin_indicator = 1 if bins[i] == bad_bin else 0
        if group_indicator and bin_indicator:
            alg.learning_rate = lr
            estimate_sign = -1 if underestimate else 1
            class_onehot = np.zeros(alg.n_actions)
            class_onehot[bad_class] = 1
            loss = np.ones(alg.n_actions) * 0.5
            loss += 0.5 * group_indicator * bin_indicator * estimate_sign * class_onehot
            alg.update(loss)
            updated.append(i)
            
    return np.array(updated)


class Adversary:
    def __init__(self, init_alg, n_actions, lr):
        if init_alg is not None:
            self.alg = init_alg(n_actions, learning_rate=lr)
        else:
            self.alg = None

    def find_bad_bin_group(self, average_labels, predictions, counts, bins, groups, lr):
        """
        Finds the bin, group, class, and whether the class is underestimated or overestimated
        that has the largest calibration error.
        Args:
            average_labels: (n_bins, n_groups, n_classes) array of average labels for each bin and group
            predictions: (n_algorithms, n_classes) array of predictions for each algorithm
            counts: (n_bins, n_groups) array of counts for each bin and group
            bins: (n_algorithms) array of bin numbers for each algorithm
            groups: (n_groups) list of lists of algorithm indices for each group
        Returns:
            (bad_bin, bad_group, bad_class, underestimate): tuple of bin, group, class, and underestimate
        """
        if self.alg is None:
            errors = calculate_calibration_errors(
                average_labels, predictions, bins, groups
            )
            weighted_errors = errors * counts[:, :, np.newaxis, np.newaxis] / len(predictions)
            return (
                np.unravel_index(weighted_errors.argmax(), weighted_errors.shape),
                weighted_errors.max(),
            )
        else:
            errors = calculate_calibration_errors(average_labels, predictions, bins, groups)
            weighted_errors = (
                errors * counts[:, :, np.newaxis, np.newaxis] / len(predictions)
            )
            self.alg.update(0.5 - 0.5 * weighted_errors.flatten())

            return (
                np.unravel_index(self.alg.predict(), weighted_errors.shape),
                weighted_errors.max(),
            )
