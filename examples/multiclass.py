import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import seaborn as sns
from collections import defaultdict
from multicalibration import MulticalibrationPredictor


def generate_correlated_subgroup_data(n_samples=1000, n_classes=3, n_features=5, n_subgroups=4, noise_level=0.1):
    """
    Generate synthetic multiclass data with correlated subgroups
    
    Args:
        n_samples: Number of samples to generate
        n_classes: Number of classes
        n_features: Number of features to generate
        n_subgroups: Number of subgroups to create
        noise_level: Amount of noise to add to predictions
    """
    # Generate feature matrix
    X = np.random.randn(n_samples, n_features)
    
    # Generate true class probabilities using a softmax of feature combinations
    true_weights = np.random.randn(n_features, n_classes)
    # true_probs = softmax(X @ true_weights, axis=1)
    
    # Generate true labels
    labels = np.random.choice(n_classes, size=n_samples, p=[1/n_classes]*n_classes)
        
    # Generate noisy predictions
    noise = np.random.randn(n_samples, n_classes) * noise_level
    noisy_logits = X @ true_weights + noise
    predicted_probs = softmax(noisy_logits, axis=1)
    
    # Create correlated subgroups
    subgroups = []
    for i in range(n_subgroups):
        # Create subgroup based on feature thresholds
        feature_idx = i % n_features
        threshold = np.random.choice([-0.5, 0, 0.5])
        if i % 2 == 0:
            subgroup = np.where(X[:, feature_idx] > threshold)[0]
        else:
            subgroup = np.where(X[:, feature_idx] <= threshold)[0]
        subgroups.append(subgroup)
    
    return predicted_probs, labels.reshape(-1), subgroups

def compute_calibration_error(probs, labels, subgroups, n_bins=10):
    """
    Compute calibration error for each subgroup and class
    
    Returns:
        dict: Calibration errors by subgroup and class
    """
    # Convert labels to one-hot encoding
    labels = np.eye(labels.max() + 1)[labels]
    n_classes = labels.shape[1]
    cal_errors = defaultdict(dict)
    
    # Compute for each subgroup
    for sg_idx, sg in enumerate(subgroups):
        if len(sg) == 0:
            continue
            
        for class_idx in range(n_classes):
            sg_probs = probs[sg, class_idx]
            sg_labels = labels[sg, class_idx]
            
            # Create bins
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(sg_probs, bins) - 1
            
            cal_error = 0
            bin_counts = np.zeros(n_bins)
            
            # Compute calibration error for each bin
            for bin_idx in range(n_bins):
                mask = bin_indices == bin_idx
                if np.sum(mask) > 0:
                    bin_counts[bin_idx] = np.sum(mask)
                    pred_prob = np.mean(sg_probs[mask])
                    true_prob = np.mean(sg_labels[mask])
                    cal_error += np.abs(pred_prob - true_prob) * (np.sum(mask) / len(sg))
            
            cal_errors[sg_idx][class_idx] = cal_error
            
    return cal_errors


def create_calibration_plots(original_probs, labels, calibrated_probs, subgroups, title, n_bins=10):
    """
    Create calibration plots comparing original and calibrated predictions
    """
    # Make labels one hot
    labels = np.eye(labels.max() + 1)[labels]
    n_classes = labels.shape[1]
    n_subgroups = len(subgroups)
    
    fig, axes = plt.subplots(n_subgroups, n_classes, figsize=(4*n_classes, 4*n_subgroups))
    if n_subgroups == 1:
        axes = axes.reshape(1, -1)
    
    for sg_idx, sg in enumerate(subgroups):
        for class_idx in range(n_classes):
            ax = axes[sg_idx, class_idx]
            
            # Get predictions and labels for this subgroup and class
            orig_probs = original_probs[sg, class_idx]
            calib_probs = calibrated_probs[sg, class_idx]
            true_labels = labels[sg, class_idx]
            
            # Create bins
            bins = np.linspace(0, 1, n_bins + 1)
            
            # Plot original predictions
            bin_indices = np.digitize(orig_probs, bins) - 1
            bin_probs = np.zeros(n_bins)
            bin_true = np.zeros(n_bins)
            for bin_idx in range(n_bins):
                mask = bin_indices == bin_idx
                if np.sum(mask) > 0:
                    bin_probs[bin_idx] = np.mean(orig_probs[mask])
                    bin_true[bin_idx] = np.mean(true_labels[mask])
            ax.scatter(bin_probs, bin_true, label='Original', alpha=0.5)
            
            # Plot calibrated predictions
            bin_indices = np.digitize(calib_probs, bins) - 1
            bin_probs = np.zeros(n_bins)
            bin_true = np.zeros(n_bins)
            for bin_idx in range(n_bins):
                mask = bin_indices == bin_idx
                if np.sum(mask) > 0:
                    bin_probs[bin_idx] = np.mean(calib_probs[mask])
                    bin_true[bin_idx] = np.mean(true_labels[mask])
            ax.scatter(bin_probs, bin_true, label='Calibrated', alpha=0.5)
            
            # Add diagonal line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            
            # Labels and title
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('True Probability')
            ax.set_title(f'Subgroup {sg_idx}, Class {class_idx}')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/{title}.png')
    plt.close()

if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 1000
    n_classes = 3
    probs, labels, subgroups = generate_correlated_subgroup_data(
        n_samples=n_samples,
        n_classes=n_classes,
        n_features=5,
        n_subgroups=4,
        noise_level=0.2
    )
    
    # Compute initial calibration error
    initial_cal_errors = compute_calibration_error(probs, labels, subgroups)
    print("\nInitial calibration errors by subgroup and class:")
    for sg_idx, class_errors in initial_cal_errors.items():
        print(f"\nSubgroup {sg_idx}:")
        for class_idx, error in class_errors.items():
            print(f"  Class {class_idx}: {error:.4f}")
    
    # Hyperparameters for HKRR predictor
    hkrr_params = {
        'alpha': 0.1,
        'lambda': 0.01,
        'max_iter': 100,
        'randomized': True,
        'use_oracle': False,
    }
    
    # Initialize and fit HKRR predictor
    mcb = MulticalibrationPredictor('HKRR', num_classes=n_classes)
    mcb.fit(probs, labels, subgroups, hkrr_params)
    
    # Make predictions using HKRR
    hkrr_probs = mcb.predict(probs, subgroups)
    
    # Compute calibration error after HKRR
    final_cal_errors = compute_calibration_error(hkrr_probs, labels, subgroups)
    print("\nFinal calibration errors by subgroup and class:")
    for sg_idx, class_errors in final_cal_errors.items():
        print(f"\nSubgroup {sg_idx}:")
        for class_idx, error in class_errors.items():
            print(f"  Class {class_idx}: {error:.4f}")
    
    # Create calibration plots
    create_calibration_plots(
        probs, labels, hkrr_probs, subgroups,
        'HKRR_Multicalibration_Results'
    )