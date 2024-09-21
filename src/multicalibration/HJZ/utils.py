import numpy as np

def calculate_average_labels(labels, bins, groups, n_bins):
    """
    This function calculates the average labels for each bin and group.
    Args:
        labels (np.array): The labels for each datapoint.
        bins (np.array): The bins for each datapoint.
        groups (list): The groups of datapoints.
        n_bins (int): The number of bins.
    Returns:
        (np.array, np.array): The true average labels for each bin and group, and the counts for each bin and group.
    """

    n_groups = len(groups)
    n_classes = labels.shape[1]
    average_labels = np.zeros((n_bins**n_classes, n_groups, n_classes))
    counts = np.zeros((n_bins**n_classes, n_groups))

    for group_idx, group in enumerate(groups):
        labels_group = labels[group]
        bins_group = bins[group]

        for unique_bin in np.unique(bins_group):
            bin_mask = bins_group == unique_bin
            average_labels[unique_bin, group_idx] = labels_group[bin_mask].mean(axis=0)
            counts[unique_bin, group_idx] = np.sum(bin_mask)

    return average_labels, counts


def discretize_values(values, n_bins, verbose=False):
    """
    This function discretizes the values into bins.
    Args:
        values (np.array): The values to discretize.
        n_bins (int): The number of bins.
        verbose (bool): Whether to return the discretized values or the bin indices.
    Returns:
        np.array: The discretized values or the bin indices.
    """

    n_datapoints, n_classes = values.shape

    bins = np.linspace(0, 1, n_bins + 1)
    bins[-1] = 1.1

    discretized_values = np.digitize(values, bins, right=False) - 1
    discretized_values[discretized_values == -1] = 0

    if verbose:
        return discretized_values
    else:
        return np.sum(discretized_values * (n_bins ** np.arange(n_classes)), axis=1)


def calculate_calibration_errors(average_labels, predictions, bins, groups):
    """
    This function calculates the calibration errors for each bin, group, and class.
    Args:
        average_labels (np.array): The true average labels for each bin and group.
        predictions (np.array): The predictions for each datapoint.
        bins (np.array): The bins for each datapoint.
        groups (list): The groups of datapoints.
    Returns:
        np.array: The calibration errors for each bin, group, and class.
    """
    n_bins, n_groups, n_classes = average_labels.shape
    errors = np.zeros(shape=(n_bins, n_groups, n_classes, 2))

    for group_idx, group in enumerate(groups):
        predictions_group = predictions[group]
        bins_group = bins[group]

        for unique_bin in np.unique(bins_group):
            bin_mask = bins_group == unique_bin
            bin_group_predictions = predictions_group[bin_mask]

            if len(bin_group_predictions) == 0:
                continue
            else:
                for k in range(n_classes):
                    prediction = bin_group_predictions[:, k].mean()
                    avg_label = average_labels[unique_bin, group_idx, k]
                    errors[unique_bin, group_idx, k] = [
                        prediction - avg_label,
                        avg_label - prediction,
                    ]

    return errors