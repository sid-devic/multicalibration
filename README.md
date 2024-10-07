# Multicalibration Post-Processing Python Package
Multicalibration is a Python package that implements a model post-processing method of the [same name](https://arxiv.org/abs/1711.08513).
The goal of multicalibration post-processing algorithms are to improve the calibration of a model not only overall, but also on specified subpopulations (or "groups"/"subgroups") given as input.
This package provides implementations of two multicalibration algorithms: HKRR (from the original multicalibration paper), and [HJZ](https://arxiv.org/abs/2302.10863).

The package can be installed via pip:
```bash
pip install multicalibration
```
The package can also be installed by cloning the git repository:
```bash
git clone https://github.com/sid-devic/multicalibration.git
cd multicalibration
pip install .
```

## Example Usage
Multicalibration post-processing takes as input a set of _probabilistic predictions_, true labels for those predictions, and a list of subgroups.
Importantly, datapoints may belong to multiple subgroups: that is, subgroups may be very complex and overlapping.
In `examples/basic_usage.py`, we give a short example of applying the HKRR algorithm on some synthetic data, summarized here.
```python
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
```
In the above code, `probs` is a length `n` array of a model's probabilistic predictions (e.g., `prob[i]` gives confidence in [0,1] that the model believes datapoint `i` should be classified with label `1`).
The `labels` array is a length `n` binary array with the true labels of each datapoint. 
Most importantly, `subgroups` is a length `n_groups` (number of subgroups) array, where each index `subgroups[j]` gives all indices of datapoints `i` which belong to subgroup `j`.
For example, if there were three datapoints and two groups, the following `subgroups` array would represent that the first two datapoints (`i=0,1`) belong to group one, and the second two datapoints (`i=1,2`) belong to group two.
```python
subgroups = [[0, 1], [1, 2]]
```

Multicalibration post-processing algorithms usually also take in a number of hyperparameters. One may want to select these hyperparameters using a hold-out validation set. We demonstrate a potential way of doing this in the file `examples/hyperparameter_search.py`.
This file also serves as an example of applying multicailbration on _real data_ from the US Census via the [folktables](https://github.com/socialfoundations/folktables) package.

## Cite
This package was developed as part of [our work](https://arxiv.org/abs/2406.06487) on the empirical aspects of multicalibration post-processing algorithms. We ask that you consider citing our paper here:
```
@article{hansen2024multicalibration,
  title={When is Multicalibration Post-Processing Necessary?},
  author={Hansen, Dutch and Devic, Siddartha and Nakkiran, Preetum and Sharan, Vatsal},
  journal={Advances in Neural Information Processing Systems (Neurips)},
  year={2024}
}
```

If you use the HKRR algorithm, we encourage you to cite the original multicalibration paper:
```
@inproceedings{hebert2018multicalibration,
  title={Multicalibration: Calibration for the (computationally-identifiable) masses},
  author={H{\'e}bert-Johnson, Ursula and Kim, Michael and Reingold, Omer and Rothblum, Guy},
  booktitle={International Conference on Machine Learning},
  pages={1939--1948},
  year={2018},
  organization={PMLR}
}
```

Finally, if you used the HJZ algorithm, please cite the authors work:
```
@article{haghtalab2024unifying,
  title={A unifying perspective on multi-calibration: Game dynamics for multi-objective learning},
  author={Haghtalab, Nika and Jordan, Michael and Zhao, Eric},
  journal={Advances in Neural Information Processing Systems (Neurips)},
  year={2023}
}
```

### Acknowledgements and License
This repository is under the MIT license. Most of the original implementation work was done by Dutch Hansen as an undergraduate research assistant at the University of Southern California.
This repository also uses Eric Zhao's implemention of HJZ from the original paper, found [here](https://github.com/ericzhao28/multicalibration) (also on the MIT license).
We sincerely thank Eric for help debugging and implementing the algorithm.
The HKRR implementation is based in part on the implementation found [here](https://github.com/sanatonek/fairness-and-callibration/tree/893c9738bf8e01d089568b1d7a56a8b53037e5fb). We thank the original authors Saina Asani, Sana Tonekaboni, and Shuja Khalid. Unfortunately, we were not able to find a license for their work.

### Contribute
Thank you for your interest!
We plan to slowly incorporate additional multicalibration-style algorithms into this package, as we believe that practitioners stand to benefit from accessible implementations of these algorithms.
Please contact `devic[at]usc.edu` if you are interested in contributing.