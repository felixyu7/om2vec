import unittest
import torch
import sys
import os

# Add project root to sys.path to allow importing project modules
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.om_processing import calculate_summary_statistics, preprocess_photon_sequence, sinusoidal_positional_encoding

class TestOMProcessing(unittest.TestCase):

    def test_calculate_summary_statistics_empty(self):
        times = torch.tensor([])
        charges = torch.tensor([])
        stats = calculate_summary_statistics(times, charges, charge_log_offset=1.0, time_log_epsilon=1.0)
        self.assertEqual(stats.shape, (9,))
        # For empty, with offset 1.0, log(1.0) = 0.0
        self.assertTrue(torch.allclose(stats, torch.zeros(9, dtype=torch.float32)))

    def test_calculate_summary_statistics_simple(self):
        times = torch.tensor([10., 20., 30.])
        charges = torch.tensor([1., 1., 1.])
        # Expected values need to be calculated manually or from a trusted source
        # For now, just checking shape and basic properties
        stats = calculate_summary_statistics(times, charges, charge_log_offset=1.0, time_log_epsilon=1.0)
        self.assertEqual(stats.shape, (9,))
        self.assertFalse(torch.isnan(stats).any())
        self.assertFalse(torch.isinf(stats).any())

    def test_preprocess_photon_sequence_empty(self):
        times = torch.tensor([])
        charges = torch.tensor([])
        max_photons = 5
        tq_log_norm_offset = 1.0
        processed_seq, mask = preprocess_photon_sequence(times, charges, max_photons, tq_log_norm_offset)
        self.assertEqual(processed_seq.shape, (max_photons, 2))
        self.assertEqual(mask.shape, (max_photons,))
        self.assertTrue(torch.all(processed_seq == 0))
        self.assertTrue(torch.all(~mask))

    def test_preprocess_photon_sequence_padding(self):
        times = torch.tensor([10., 20.])
        charges = torch.tensor([1., 1.])
        max_photons = 5
        tq_log_norm_offset = 1.0
        processed_seq, mask = preprocess_photon_sequence(times, charges, max_photons, tq_log_norm_offset)
        
        self.assertEqual(processed_seq.shape, (max_photons, 2))
        self.assertEqual(mask.shape, (max_photons,))
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])
        self.assertFalse(mask[2])
        # Check normalized values (log(x+1))
        # log(10+1) = log(11) approx 2.397
        # log(1+1) = log(2) approx 0.693
        self.assertAlmostEqual(processed_seq[0, 0].item(), torch.log(times[0] + tq_log_norm_offset).item(), places=4)
        self.assertAlmostEqual(processed_seq[0, 1].item(), torch.log(charges[0] + tq_log_norm_offset).item(), places=4)

    def test_preprocess_photon_sequence_truncation(self):
        times = torch.tensor([10., 20., 30., 40., 50., 60.])
        charges = torch.tensor([1., 1., 1., 1., 1., 1.])
        max_photons = 3
        tq_log_norm_offset = 1.0
        # Ensure sorted for predictable truncation
        sorted_indices = torch.argsort(times)
        sorted_times = times[sorted_indices]
        sorted_charges = charges[sorted_indices]

        processed_seq, mask = preprocess_photon_sequence(sorted_times, sorted_charges, max_photons, tq_log_norm_offset)
        self.assertEqual(processed_seq.shape, (max_photons, 2))
        self.assertEqual(mask.shape, (max_photons,))
        self.assertTrue(torch.all(mask))
        self.assertAlmostEqual(processed_seq[0, 0].item(), torch.log(sorted_times[0] + tq_log_norm_offset).item(), places=4)
        self.assertAlmostEqual(processed_seq[-1, 0].item(), torch.log(sorted_times[max_photons-1] + tq_log_norm_offset).item(), places=4)


    def test_sinusoidal_positional_encoding(self):
        max_len = 10
        d_model = 16
        pe = sinusoidal_positional_encoding(max_len, d_model)
        self.assertEqual(pe.shape, (max_len, d_model))
        # Check some properties - e.g., values are within [-1, 1]
        self.assertTrue(torch.all(pe >= -1.0) and torch.all(pe <= 1.0))

if __name__ == '__main__':
    unittest.main()