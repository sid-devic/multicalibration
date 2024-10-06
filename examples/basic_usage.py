from multicalibration import MulticalibrationPredictor
from utils import generate_correlated_subgroup_data, create_calibration_plots, calibration_error


if __name__ == '__main__':
    # Generate some synthetic data
    probs, labels, subgroups = generate_correlated_subgroup_data(n_samples=1000)
    n_groups = len(subgroups)

    # Hyperparams for HKRR predictor
    hkrr_params = {
        'alpha': 0.1,
        'lambda': 0.01,
        'max_iter': 100,
        'randomized': True,
        'use_oracle': False,
    }

    # Initialize and fit HKRR predictor
    mcb = MulticalibrationPredictor('HKRR')
    mcb.fit(probs, labels, subgroups, hkrr_params)

    # Make predictions using HKRR
    hkrr_probs = mcb.predict(probs, subgroups)

    # Create calibration plots for HKRR
    create_calibration_plots(probs, labels, hkrr_probs, subgroups, 'HKRR_Multicalibration')

    # Updated parameters for HJZ predictor
    hjz_params = {
        'iterations': 200,
        'algorithm': 'OptimisticHedge',
        'other_algorithm': 'OptimisticHedge',
        'lr': 0.995,
        'other_lr': 0.995,
        'n_bins': 10,
    }

    # Initialize and fit HJZ predictor
    hjz_predictor = MulticalibrationPredictor('HJZ')
    hjz_predictor.fit(probs, labels, subgroups, hjz_params)

    # Make predictions using HJZ
    hjz_probs = hjz_predictor.predict(probs, subgroups)

    # Create calibration plots for HJZ
    create_calibration_plots(probs, labels, hjz_probs, subgroups, 'HJZ_Multicalibration')

    print("\nCalibration Errors (ECE):")
    print(f"Original: {calibration_error(labels, probs):.4f}")
    print(f"HKRR: {calibration_error(labels, hkrr_probs):.4f}")
    print(f"HJZ: {calibration_error(labels, hjz_probs):.4f}")

    # Per-group calibration errors
    print("\nPer-group Calibration Errors (ECE):")
    for i in range(n_groups):
        group_mask = subgroups[i]
        print(f"\nGroup {i+1}:")
        print(f"Original: {calibration_error(labels[group_mask], probs[group_mask]):.4f}")
        print(f"HKRR: {calibration_error(labels[group_mask], hkrr_probs[group_mask]):.4f}")
        print(f"HJZ: {calibration_error(labels[group_mask], hjz_probs[group_mask]):.4f}")