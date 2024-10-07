from multicalibration import MulticalibrationPredictor
from utils import calibration_error, convert_groups, worst_group_calibration_error
from sklearn.model_selection import ParameterGrid, train_test_split
from folktables import ACSDataSource, ACSIncome
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
import numpy as np
import os


def load_and_fit_NB():
    """
    Load ACSIncome data, train a Naive Bayes model on the data, and return the predictions, labels, and groups
        for each split.
    """
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
    model = GaussianNB()

    # Train NB on CA data
    print('Training naive bayes base model...')
    model.fit(train_features, train_labels)

    # Get predictions for each split
    train_probs = model.predict_proba(train_features)[:, 1]
    val_probs = model.predict_proba(val_features)[:, 1]
    test_probs = model.predict_proba(test_features)[:, 1]

    return (train_probs, train_labels, train_groups), (val_probs, val_labels, val_groups), (test_probs, test_labels, test_groups)


if __name__ == '__main__':
    # Make directory for plots if it doesn't exist
    os.makedirs('plots', exist_ok=True) 

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

    _, val, test = load_and_fit_NB()
    probs_val, y_val, groups_val = val
    probs_test, y_test, groups_test = test

    # Use 20% of the validation set as holdout to evaluate the model
    probs_val_train, probs_val_holdout, y_val_train, y_val_holdout, groups_val_train, groups_val_holdout = train_test_split(probs_val, y_val, groups_val, test_size=0.2, random_state=42)

    # Only consider groups which have at least one data point in the val and test sets
    valid_group_list = list(set(groups_val) & set(groups_test))

    # Groups is currently a single list, where each index represents the group of the corresponding data point. 
    # However, we need to convert this to a list of lists where each entry is a list of all indices of data belonging to a certain subgroup.
    subgroups_val_train = convert_groups(groups_val_train, valid_group_list)
    subgroups_val_holdout = convert_groups(groups_val_holdout, valid_group_list)
    subgroups_test = convert_groups(groups_test, valid_group_list)

    # Initialize and fit HKRR predictor for each param setting in param_settings_list
    hkrr_val_cal_errors = []
    hkrr_val_worst_group_errors = []
    hkrr_objects = []
    for params in param_settings_list:
        mcb = MulticalibrationPredictor('HKRR', verbose=True)

        # Fit the model on the validation train set
        mcb.fit(probs_val_train, y_val_train, subgroups_val_train, params)
        hkrr_probs = mcb.predict(probs_val_holdout, subgroups_val_holdout)

        # Append the calibration error and multicalibration error on the validation holdout set
        hkrr_val_cal_errors.append(calibration_error(y_val_holdout, hkrr_probs))
        hkrr_val_worst_group_errors.append(worst_group_calibration_error(y_val_holdout, hkrr_probs, subgroups_val_holdout))
        hkrr_objects.append(mcb)
    
    # Select two models, one with the lowest holdout calibration error and one with 
    # the lowest holdout worst group calibration error
    best_calib_model = hkrr_objects[np.argmin(hkrr_val_cal_errors)]
    best_multicalib_model = hkrr_objects[np.argmin(hkrr_val_worst_group_errors)]

    # Get the test set calibration errors for the best models
    best_calib_probs = best_calib_model.predict(probs_test, subgroups_test)
    best_multicalib_probs = best_multicalib_model.predict(probs_test, subgroups_test)
    best_calib_error = calibration_error(y_test, best_calib_probs)
    best_multicalib_error = calibration_error(y_test, best_multicalib_probs)

    print('Baseline test calibration error:', calibration_error(y_test, probs_test))
    print('Calibration test error of model with lowest holdout calibration error:', best_calib_error)
    print('Multicalibration test error of model with lowest holdout multicalibration error:', best_multicalib_error)

    # --------
    # The rest of the code is for plotting the test calibration errors across different groups.
    plt.style.use('seaborn')

    # Define colors
    CB_BLUE = '#3752B8'
    CB_GREEN = '#4DAf4A'
    CB_PURPLE = '#904E9A'
    CB_ORANGE = '#FF7F00'

    # Set the width of each bar and the positions for the groups
    bar_width = 0.25
    r1 = np.arange(len(subgroups_test))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), height_ratios=[3, 1])

    # Upper subplot: Calibration Errors
    ax1.bar(r1, [calibration_error(y_test[group], probs_test[group]) for group in subgroups_test], 
        width=bar_width, label='Baseline', color=CB_BLUE, edgecolor='grey')
    ax1.bar(r2, [calibration_error(y_test[group], best_calib_probs[group]) for group in subgroups_test], 
        width=bar_width, label='Select Params on Calibration Error', color=CB_GREEN, edgecolor='grey')
    ax1.bar(r3, [calibration_error(y_test[group], best_multicalib_probs[group]) for group in subgroups_test], 
        width=bar_width, label='Select Params on Multicalibration Error', color=CB_PURPLE, edgecolor='grey')

    # Customize upper subplot
    ax1.set_ylabel('Expected Calibration Error (ECE)', fontsize=12)
    ax1.set_title('Test Calibration Errors Across Different Groups', fontsize=14, fontweight='bold')
    ax1.set_xticks([r + bar_width for r in range(len(subgroups_test))])
    ax1.set_xticklabels([])  # Remove x-axis labels from upper subplot
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of each bar in upper subplot
    for i, bars in enumerate([ax1.containers[0], ax1.containers[1], ax1.containers[2]]):
        ax1.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)

    # Customize the legend for upper subplot
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

    # Lower subplot: Subgroup Sizes
    subgroup_sizes = [len(y_test[group]) for group in subgroups_test]
    total_size = sum(subgroup_sizes)
    relative_sizes = [size / total_size for size in subgroup_sizes]

    ax2.bar(r1, relative_sizes, width=bar_width*3, color=CB_ORANGE, edgecolor='grey')

    # Customize lower subplot
    ax2.set_xlabel('Group', fontsize=12)
    ax2.set_ylabel('Relative Size', fontsize=12)
    ax2.set_title('Relative Sizes of Subgroups in Test Set', fontsize=12)
    ax2.set_xticks([r + bar_width for r in range(len(subgroups_test))])
    ax2.set_xticklabels(valid_group_list, rotation=45, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Add percentage labels on top of each bar in lower subplot
    for i, v in enumerate(relative_sizes):
        ax2.text(i, v, f'{v:.1%}', ha='center', va='bottom')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig('plots/test_calibration_errors_with_sizes.png')