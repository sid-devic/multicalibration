from multicalibration import MulticalibrationPredictor
from utils import calibration_error
from sklearn.model_selection import ParameterGrid, train_test_split
from folktables import ACSDataSource, ACSIncome
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import numpy as np


def convert_groups(group_list, group_set):
    """
    Convert a list of group indices to a list of lists of indices, where each entry is a list of all indices of data belonging to a certain subgroup.
    """
    subgroups = []
    for i in group_set:
        # Indices of datapoints belonging to subgroup i in numpy
        subgroup_indices = np.where(group_list == i)[0]
        subgroups.append(subgroup_indices)

    return subgroups


def worst_group_calibration_error(probs, labels, groups):
    """
    Calculate the calibration error for the worst-performing group.
    """
    group_errors = []
    for group in groups:
        group_errors.append(calibration_error(labels[group], probs[group]))

    return max(group_errors)    


def grouped_train_test_split(X, y, groups, test_size=0.2, random_state=None):
    """
    Split data into train and test sets, ensuring that each group is represented in both sets. This only works because each data point belongs to exactly one group (not true in general!).
    """
    train_indices = []
    test_indices = []
    groups_train = []
    groups_test = []
    for group in groups:
        group_train_indices, group_test_indices = train_test_split(group, test_size=test_size, random_state=random_state)
        train_indices.extend(group_train_indices)
        test_indices.extend(group_test_indices)
        groups_train.append(group_test_indices)
        groups_test.append(group_test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices], groups_train, groups_test


def load_and_fit_LR():
    # Load ACS data
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    ca_data = data_source.get_data(states=["CA"], download=True)
    features, labels, groups = ACSIncome.df_to_numpy(ca_data)

    # Data pre-processing
    # Scale features
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    # Convert labels to {0, 1} from {False, True}
    labels = labels.astype(int)
    # Split into train, test, validation
    train_features, test_features, train_labels, test_labels, train_groups, test_groups = train_test_split(features, labels, groups, test_size=0.2, random_state=42)
    train_features, val_features, train_labels, val_labels, train_groups, val_groups = train_test_split(train_features, train_labels, train_groups, test_size=0.2, random_state=42)
    model = LogisticRegression()

    # Train on CA data
    print('Training linear regression model...')
    model.fit(train_features, train_labels)

    # Get predictions for each split
    train_probs = model.predict_proba(train_features)[:, 1]
    val_probs = model.predict_proba(val_features)[:, 1]
    test_probs = model.predict_proba(test_features)[:, 1]

    # Groups is currently a single list, where each index represents the group of the corresponding data point. However, we need to convert this to a list of lists where each entry is a list of all indices of data belonging to a certain subgroup.
    group_set = set(groups)

    return (train_probs, train_labels, convert_groups(train_groups, group_set)), (val_probs, val_labels, convert_groups(val_groups, group_set)), (test_probs, test_labels, convert_groups(test_groups, group_set))


if __name__ == '__main__':
    # Example of a simple hyperparameter search over hkrr_params using sklearn ParameterGrid
    # We use the ACSIncome dataset from the folktables package as an example.
    param_grid = {
        'alpha': [0.05, 0.1, 0.2],
        'lambda': [0.01, 0.05, 0.1],
        'max_iter': [5, 10, 100],
        'randomized': [True, False],
        'use_oracle': [True, False]
    }

    param_settings_list = list(ParameterGrid(param_grid))

    _, val, test = load_and_fit_LR()
    probs_val, y_val, groups_val = val
    probs_test, y_test, groups_test = test

    # Only consider groups which have at least one data point in the validation set
    groups_in_val_set = [idx for idx in range(len(groups_val)) if len(groups_val[idx]) > 0]
    groups_val = [groups_val[idx] for idx in groups_in_val_set]
    groups_test = [groups_test[idx] for idx in groups_in_val_set]

    # Initialize and fit HKRR predictor for each param setting in param_settings_list
    hkrr_val_cal_errors = []
    hkrr_val_worst_group_errors = []
    hkrr_objects = []

    for params in param_settings_list:
        mcb = MulticalibrationPredictor('HKRR')
        # Use 20% of the validation set as holdout to evaluate the model
        probs_val_train, probs_val_holdout, y_val_train, y_val_holdout, groups_val_train, groups_val_holdout = grouped_train_test_split(probs_val, y_val, groups_val, test_size=0.2, random_state=42)

        # Fit the model on the training set
        mcb.fit(probs_val_train, y_val_train, groups_val_train, params)
        hkrr_probs = mcb.predict(probs_val_holdout, groups_val_holdout)
        hkrr_val_cal_errors.append(calibration_error(y_val_holdout, hkrr_probs))
        hkrr_val_worst_group_errors.append(worst_group_calibration_error(hkrr_probs, y_val_holdout, groups_val_holdout))
        hkrr_objects.append(mcb)
    
    # Select two models, one with the lowest validation calibration error and one with the lowest worst group calibration error
    best_val_model = hkrr_objects[np.argmin(hkrr_val_cal_errors)]
    best_worst_group_model = hkrr_objects[np.argmin(hkrr_val_worst_group_errors)]

    val_calib_errors = []
    
    for group in groups_val:
        if len(group) == 0:
            continue
        print('probs_val group', probs_val[group])
        val_calib_errors.append(calibration_error(y_val[group], probs_val[group]))
    # Plot the calibration errors for each group in the validation set
    plt.bar(range(len(val_calib_errors)), val_calib_errors)
    plt.xlabel('Group')
    plt.ylabel('ECE')
    plt.title('Validation Calibration Errors')
    plt.show()

    # # Initialize and fit HKRR predictor
    # mcb = MulticalibrationPredictor('HKRR')
    # mcb.fit(probs, labels, subgroups, hkrr_params)

    # # Make predictions using HKRR
    # hkrr_probs = mcb.predict(probs, subgroups)

    # # Create calibration plots for HKRR
    # create_calibration_plots(probs, labels, hkrr_probs, subgroups, 'HKRR_Multicalibration')

    # print("\nCalibration Errors (ECE):")
    # print(f"Original: {calibration_error(labels, probs):.4f}")
    # print(f"HKRR: {calibration_error(labels, hkrr_probs):.4f}")

    # # Per-group calibration errors
    # print("\nPer-group Calibration Errors (ECE):")
    # for i in range(n_groups):
    #     group_mask = subgroups[i]
    #     print(f"\nGroup {i+1}:")
    #     print(f"Original: {calibration_error(labels[group_mask], probs[group_mask]):.4f}")
    #     print(f"HKRR: {calibration_error(labels[group_mask], hkrr_probs[group_mask]):.4f}")
