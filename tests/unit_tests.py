import unittest
import numpy as np
from multicalibration.mcb import MulticalibrationPredictor
from multicalibration.HKRR.hkrr import HKRRAlgorithm
from multicalibration.HJZ.hjz import HJZAlgorithm

class TestMulticalibrationPredictor(unittest.TestCase):
    def setUp(self):
        self.hkrr_params = {
            'alpha': 0.1,
            'lambda': 0.01,
            'use_oracle': True,
            'randomized': True,
            'max_iter': 100
        }
        self.hjz_params = {
            'iterations': 100,
            'algorithm': 'Hedge',
            'other_algorithm': 'OptimisticHedge',
            'lr': 0.9,
            'other_lr': 0.9,
            'n_bins': 10,
        }

    def test_initialization(self):
        hkrr_predictor = MulticalibrationPredictor('HKRR', self.hkrr_params)
        self.assertIsInstance(hkrr_predictor.mcbp, HKRRAlgorithm)

        hjz_predictor = MulticalibrationPredictor('HJZ', self.hjz_params)
        self.assertIsInstance(hjz_predictor.mcbp, HJZAlgorithm)

        with self.assertRaises(ValueError):
            MulticalibrationPredictor('INVALID', {})

    def test_fit_input_validation(self):
        predictor = MulticalibrationPredictor('HKRR', self.hkrr_params)
        
        # Test binary labels
        with self.assertRaises(ValueError):
            predictor.fit(np.array([0.1, 0.2, 0.3]), np.array([0, 1, 2]), np.array([[1, 0], [0, 1], [1, 1]]))

    def test_fit_and_predict(self):
        predictor = MulticalibrationPredictor('HKRR', self.hkrr_params)
        
        # Sample data
        confs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1, 1])
        subgroups = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1]])

        predictor.fit(confs, labels, subgroups)

        # Test batch_predict
        f_xs = np.array([0.2, 0.6, 0.8])
        groups = np.array([[1, 0], [0, 1], [1, 1]])
        predictions = predictor.batch_predict(f_xs, groups)

        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(0 <= p <= 1 for p in predictions))


if __name__ == '__main__':
    unittest.main()