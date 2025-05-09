import unittest
import torch
import sys
import os
import yaml

# Add project root to sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.om2vec_model import Om2vecModel, FiLMLayer

# Minimal dummy config for testing
DUMMY_CONFIG_STR = """
project_name: "om2vec_test"
project_save_dir: "./experiment_logs_test"
training: true
accelerator: "cpu"
num_devices: 1
dataloader: "dummy_loader"
model_name: "om2vec_model"

data_options:
  max_photons_per_om: 64 # Smaller for faster tests
  tq_log_norm_offset: 1.0
  sensor_pos_norm_scale: 600.0
  time_log_epsilon: 1.0 # For summary stats if it uses it

model_options:
  input_embedding_dim: 32
  transformer_hidden_dim: 64 # Smaller for tests
  transformer_num_heads: 2
  transformer_num_layers: 1
  transformer_dropout: 0.0
  pooling_strategy: "mean"
  latent_learned_dim: 8
  sensor_pos_embedding_dim: 16
  sensor_integration_type: "film" # Test with film
  
  cnf_base_dist_dim: 1
  cnf_num_layers: 2
  cnf_hidden_dims_hypernet: [32, 32]
  cnf_num_bins_spline: 5
  # cnf_condition_on_sensor_pos: false # This is now default / removed

training_options:
  lr: 0.0001
  kl_beta_initial_value: 0.0
  kl_beta_final_value: 1.0
  kl_anneal_epochs: 1
"""
DUMMY_CONFIG = yaml.safe_load(DUMMY_CONFIG_STR)

class TestOm2vecModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = DUMMY_CONFIG
        cls.model = Om2vecModel(cls.cfg)
        cls.model.eval() # Set to eval mode for deterministic outputs where applicable

        # Common dimensions for tests
        cls.batch_size = 2
        cls.num_valid_oms_per_event = [2, 1] # e.g. event 0 has 2 OMs, event 1 has 1 OM
        cls.total_valid_oms = sum(cls.num_valid_oms_per_event)
        cls.max_photons = cls.cfg['data_options']['max_photons_per_om']
        cls.latent_learned_dim = cls.cfg['model_options']['latent_learned_dim']
        cls.summary_dim = 9

    def _generate_dummy_input_for_encode_decode(self, num_oms):
        raw_tq = torch.rand(num_oms, self.max_photons, 2) * 100 # times and charges
        # Create some actual hits
        hit_masks = torch.zeros(num_oms, self.max_photons, dtype=torch.bool)
        for i in range(num_oms):
            num_actual_hits_this_om = torch.randint(1, self.max_photons + 1, (1,)).item()
            hit_masks[i, :num_actual_hits_this_om] = True
        
        # Ensure charges are non-negative for log
        raw_tq[..., 1].clamp_(min=0.0)

        sensor_pos = torch.rand(num_oms, 3) * 100
        return raw_tq, hit_masks, sensor_pos

    def _generate_dummy_batch_for_forward(self):
        # Simulates output from dataloader collate_fn
        max_oms_in_event = max(self.num_valid_oms_per_event) if self.num_valid_oms_per_event else 0
        
        all_om_hits = torch.zeros(self.batch_size, max_oms_in_event, self.max_photons, 2)
        all_om_sensor_pos = torch.zeros(self.batch_size, max_oms_in_event, 3)
        om_mask = torch.zeros(self.batch_size, max_oms_in_event, dtype=torch.bool)
        hit_mask = torch.zeros(self.batch_size, max_oms_in_event, self.max_photons, dtype=torch.bool)

        current_om_idx = 0
        for i_event in range(self.batch_size):
            num_oms_this_event = self.num_valid_oms_per_event[i_event] if i_event < len(self.num_valid_oms_per_event) else 0
            if num_oms_this_event > 0:
                om_mask[i_event, :num_oms_this_event] = True
                # Generate data for these OMs
                tq_event, hm_event, sp_event = self._generate_dummy_input_for_encode_decode(num_oms_this_event)
                all_om_hits[i_event, :num_oms_this_event, :, :] = tq_event
                all_om_sensor_pos[i_event, :num_oms_this_event, :] = sp_event
                hit_mask[i_event, :num_oms_this_event, :] = hm_event
        
        return {
            "all_om_hits": all_om_hits,
            "all_om_sensor_pos": all_om_sensor_pos,
            "om_mask": om_mask,
            "hit_mask": hit_mask
        }

    def test_model_instantiation(self):
        self.assertIsInstance(self.model, Om2vecModel)

    def test_film_layer(self):
        context_dim = self.cfg['model_options']['sensor_pos_embedding_dim']
        feature_dim = self.cfg['model_options']['input_embedding_dim']
        film = FiLMLayer(context_dim, feature_dim)
        
        N_valid = self.total_valid_oms
        P = self.max_photons
        
        features = torch.rand(N_valid, P, feature_dim)
        context = torch.rand(N_valid, context_dim)
        
        modulated_features = film(features, context)
        self.assertEqual(modulated_features.shape, features.shape)

    def test_encode_basic(self):
        raw_tq, hit_masks, sensor_pos = self._generate_dummy_input_for_encode_decode(self.total_valid_oms)
        
        # Test with return_dist_params = False (should return sampled z_learned or mu_learned)
        self.model.eval() # Ensure mu_learned is returned
        z_summary, z_learned_repr = self.model.encode(raw_tq, hit_masks, sensor_pos, return_dist_params=False)
        
        self.assertEqual(z_summary.shape, (self.total_valid_oms, self.summary_dim))
        self.assertEqual(z_learned_repr.shape, (self.total_valid_oms, self.latent_learned_dim))
        self.assertFalse(torch.isnan(z_summary).any())
        self.assertFalse(torch.isnan(z_learned_repr).any())

    def test_encode_return_dist_params(self):
        raw_tq, hit_masks, sensor_pos = self._generate_dummy_input_for_encode_decode(self.total_valid_oms)
        
        z_summary, mu_learned, log_sigma_sq_learned = self.model.encode(raw_tq, hit_masks, sensor_pos, return_dist_params=True)
        
        self.assertEqual(z_summary.shape, (self.total_valid_oms, self.summary_dim))
        self.assertEqual(mu_learned.shape, (self.total_valid_oms, self.latent_learned_dim))
        self.assertEqual(log_sigma_sq_learned.shape, (self.total_valid_oms, self.latent_learned_dim))

    def test_decode_with_times_to_evaluate(self):
        z_summary = torch.rand(self.total_valid_oms, self.summary_dim)
        z_learned = torch.rand(self.total_valid_oms, self.latent_learned_dim)
        num_eval_times = 10
        times_to_eval = torch.linspace(0, 1000, num_eval_times) # (num_eval_times,)

        eval_times, log_pdf = self.model.decode(z_summary, z_learned, times_to_evaluate=times_to_eval)
        
        self.assertEqual(eval_times.shape, (self.total_valid_oms, num_eval_times))
        self.assertEqual(log_pdf.shape, (self.total_valid_oms, num_eval_times))
        self.assertFalse(torch.isnan(log_pdf).any())

    def test_decode_with_bins_and_range(self):
        z_summary = torch.rand(self.total_valid_oms, self.summary_dim)
        z_learned = torch.rand(self.total_valid_oms, self.latent_learned_dim)
        num_time_bins = 20
        time_range = (0., 2000.)

        eval_times, log_pdf = self.model.decode(z_summary, z_learned, num_time_bins=num_time_bins, time_range=time_range)
        
        self.assertEqual(eval_times.shape, (self.total_valid_oms, num_time_bins))
        self.assertEqual(log_pdf.shape, (self.total_valid_oms, num_time_bins))

    def test_forward_pass_basic(self):
        dummy_batch = self._generate_dummy_batch_for_forward()
        outputs = self.model.forward(dummy_batch)
        
        self.assertIn("reconstruction_log_prob", outputs)
        self.assertIn("kl_divergence", outputs)
        self.assertIn("num_valid_oms", outputs)
        self.assertEqual(outputs["num_valid_oms"], self.total_valid_oms)
        self.assertIsNotNone(outputs["reconstruction_log_prob"])
        self.assertFalse(torch.isnan(outputs["reconstruction_log_prob"]).any())
        self.assertFalse(torch.isnan(outputs["kl_divergence"]).any())

    def test_forward_pass_no_valid_oms(self):
        # Create a batch where om_mask is all False
        dummy_batch = self._generate_dummy_batch_for_forward()
        dummy_batch["om_mask"] = torch.zeros_like(dummy_batch["om_mask"], dtype=torch.bool)
        
        outputs = self.model.forward(dummy_batch)
        self.assertEqual(outputs["num_valid_oms"], 0)
        self.assertTrue(torch.allclose(outputs["reconstruction_log_prob"], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(outputs["kl_divergence"], torch.tensor(0.0)))

    def test_shared_step(self):
        # This indirectly tests forward as well
        dummy_batch = self._generate_dummy_batch_for_forward()
        self.model.training = True # To ensure KL beta is applied if configured
        loss = self.model._shared_step(dummy_batch, 0, 'train')
        self.assertIsNotNone(loss)
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())
        self.model.eval()


if __name__ == '__main__':
    unittest.main()