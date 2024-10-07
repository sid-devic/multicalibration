import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import os


# Set a nicer style for plots
plt.style.use('seaborn')


def generate_correlated_subgroup_data(n_samples=1000):
    """
    Generate sample data where group membership is correlated with the label.
    """
    n_groups = 4

    # Generate sample data with more uncalibrated probabilities
    np.random.seed(42)
    true_probs = np.random.beta(2, 5, n_samples)  # True probabilities
    confs = np.power(true_probs, 0.3)  # Uncalibrated predictions (overconfident)
    labels = np.random.binomial(1, true_probs)

    # Correlated subgroups: Each group assignment depends on the label
    group_membership = np.zeros((n_samples, n_groups))

    for i in range(n_groups):
        # Each subgroup is correlated with the label but allows for membership in multiple groups
        subgroup_prob = 0.6 * labels + 0.3 * np.random.rand(n_samples)  # Higher probability of being in group if label == 1
        group_membership[:, i] = np.random.binomial(1, subgroup_prob)

    # Convert the group membership matrix to a list of lists of indices
    subgroups = []
    for i in range(n_groups):
        subgroup_indices = np.where(group_membership[:, i] == 1)[0]  # Indices of datapoints belonging to subgroup i
        subgroups.append(subgroup_indices.tolist())  # Convert to list and append

    # Add in overall group
    subgroups.append(list(range(n_samples)))

    # Optional: print the first few subgroup indices for verification
    for i, subgroup in enumerate(subgroups):
        print(f"Subgroup {i}: {subgroup[:10]} (first 10 indices)")

    # Optional: print summary statistics
    print(f"Subgroup mean correlation with labels: {np.corrcoef(group_membership.T, labels)[-1, :-1]}")
    print(f"Number of individuals in multiple subgroups: {(group_membership.sum(axis=1) > 1).sum()}")
    print('confs', confs)
    print('labels', len(labels), 'number of labels', len(np.unique(labels, return_counts=True)))
    print('subgroups', len(subgroups))
    print('size of each subgroup', [len(subgroup) for subgroup in subgroups])

    return confs, labels, subgroups


def plot_calibration(ax, y_true, y_pred, label, color):
    """
    Plot calibration curve for a given set of true labels and predicted probabilities.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy='quantile')
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label=label, color=color, markersize=4)


def calibration_error(y_true, y_pred, n_bins=10):
    """
    Calculate the expected calibration error (ECE) for a set of true labels and predicted probabilities.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='quantile')
    return np.mean(np.abs(prob_true - prob_pred))


def create_calibration_plots(original_confs, labels, predictions, subgroups, method, n_groups=4):
    """
    Create calibration plots for the overall dataset and each subgroup.
    """
    # Make directory for plots if it doesn't exist
    os.makedirs('plots', exist_ok=True) 

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Overall calibration
    ax = axes[0, 0]
    plot_calibration(ax, labels, original_confs, 'Original', 'blue')
    plot_calibration(ax, labels, predictions, 'Calibrated', 'red')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Overall Calibration', fontsize=10)
    ax.legend(loc='lower right', fontsize=8)

    # Calibration for each group
    for i in range(n_groups):
        if i < 3:  # We only have space for 3 group plots
            row = (i + 1) // 2
            col = (i + 1) % 2
            ax = axes[row, col]
            group_mask = subgroups[i]
            plot_calibration(ax, labels[group_mask], original_confs[group_mask], 'Original', 'blue')
            plot_calibration(ax, labels[group_mask], predictions[group_mask], 'Calibrated', 'red')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
            ax.set_xlabel('Mean predicted probability')
            ax.set_ylabel('Fraction of positives')
            ax.set_title(f'Calibration for Group {i+1}', fontsize=10)
            ax.legend(loc='lower right', fontsize=8)

    plt.suptitle(method, fontsize=14)
    # plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    plt.savefig(f'plots/{method}.png')


def convert_groups(group_list, group_set):
    """
    Convert a list of group indices to a list of lists of indices, where each entry is a list of all indices of 
        data belonging to a certain subgroup.
    """
    subgroups = []
    for i in group_set:
        # Indices of datapoints belonging to subgroup i in numpy
        subgroup_indices = np.where(group_list == i)[0]
        subgroups.append(subgroup_indices)

    return subgroups


def worst_group_calibration_error(labels, probs, groups):
    """
    Calculate the calibration error for the worst-performing group.
    """
    group_errors = []
    for group in groups:
        group_errors.append(calibration_error(labels[group], probs[group]))

    return max(group_errors)