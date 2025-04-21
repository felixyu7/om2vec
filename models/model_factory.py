import lightning.pytorch as pl
import yaml, os
import torch

# Import renamed model classes
from .transformer_ae import TransformerAE
from .fc_ae import FCAE
from .tcn_ae import TCNAE

def get_model(cfg: dict, dataset_size: int | None, cfg_path: str | None = None) -> pl.LightningModule:
    """Factory function to instantiate or load an AE/VAE model based on config.

    Args:
        cfg: Dictionary containing the configuration.
        dataset_size: Size of the training dataset (needed for beta annealing schedule).
                      Set to None or -1 for inference/non-training.
        cfg_path: Optional path to the config file. Used to infer architecture 
                  if 'architecture' key is missing in cfg['model_options'].

    Returns:
        An initialized PyTorch Lightning model.

    Raises:
        ValueError: If the architecture is unknown or cannot be determined.
        KeyError: If required configuration keys are missing.
    """
    model_options = cfg.get('model_options', {}) 
    training_options = cfg.get('training_options', {})
    checkpoint_path = cfg.get('checkpoint', '')
    
    # --- Determine Architecture --- 
    architecture = model_options.get('architecture')
    if not architecture and cfg_path:
        fname = os.path.basename(cfg_path).lower()
        if 'transformer' in fname:
            architecture = 'transformer'
        elif 'fc' in fname:
            architecture = 'fc'
        elif 'tcn' in fname:
            architecture = 'tcn'
        else:
             print("Warning: Could not infer architecture from config filename. Please specify 'model_options.architecture'. Defaulting to 'transformer'.")
             architecture = 'transformer' # Default if inference fails
    elif not architecture:
         # Try to get architecture from the checkpoint if loading one
         if checkpoint_path and os.path.exists(checkpoint_path):
             try:
                 ckpt = torch.load(checkpoint_path, map_location='cpu')
                 if 'hyper_parameters' in ckpt and 'architecture' in ckpt['hyper_parameters']:
                     architecture = ckpt['hyper_parameters']['architecture']
                     print(f"Inferred architecture '{architecture}' from checkpoint.")
                 else:
                     raise ValueError("Architecture not found in checkpoint hyperparameters.")
             except Exception as e:
                 print(f"Warning: Could not load or infer architecture from checkpoint {checkpoint_path}: {e}")
                 raise ValueError("Model architecture must be specified in config ('model_options.architecture') or inferable from config/checkpoint.") from e
         else:
              raise ValueError("Model architecture must be specified in config ('model_options.architecture') or inferable from config filename.")

    print(f"Selected architecture: {architecture}")

    # --- Prepare Model Arguments --- 
    # Common args for all models
    is_vae = model_options.get('is_vae', False) # Default to AE
    beta_schedule = model_options.get('beta_schedule', 'none')
    if is_vae and beta_schedule == 'none':
         print("Warning: Model set to VAE (is_vae=True) but beta_schedule is 'none'. KL term will be zero.")
    
    model_args = {
        'latent_dim': model_options.get('latent_dim', 64),
        'is_vae': is_vae,
        'beta_schedule': beta_schedule if is_vae else 'none', # Force 'none' if not VAE
        'beta_factor': model_options.get('beta_factor', 1e-5),
        'beta_peak_epoch': model_options.get('beta_peak_epoch', 4),
        'dataset_size': dataset_size if dataset_size is not None else -1,
        'batch_size': training_options.get('batch_size', 128),
        'lr': training_options.get('lr', 1e-3),
        'lr_schedule': training_options.get('lr_schedule', None),
        'weight_decay': training_options.get('weight_decay', 1e-5)
    }

    # Architecture-specific args
    if architecture == 'transformer':
        ModelClass = TransformerAE
        model_args.update({
            'in_features': model_options.get('in_features', 6400),
            'embed_dim': model_options.get('embed_dim', 32),
            'sensor_positional_encoding': model_options.get('sensor_positional_encoding', False),
        })
    elif architecture == 'fc':
        ModelClass = FCAE
        model_args.update({
            'in_features': model_options.get('in_features', 6400),
            'fc_hidden_dims': model_options.get('fc_hidden_dims', [1024, 512]),
            # Note: beta factor/peak handled by common args
        })
    elif architecture == 'tcn':
        ModelClass = TCNAE
        model_args.update({
            'seq_len': model_options.get('seq_len', model_options.get('in_features', 6400)), # Use seq_len or in_features
            'tcn_channels': model_options.get('tcn_channels', [64, 64, 64, 64]),
            'tcn_kernel_size': model_options.get('tcn_kernel_size', 3),
            'tcn_dropout': model_options.get('tcn_dropout', 0.1),
            'teacher_forcing_ratio': training_options.get('teacher_forcing_ratio', 0.5) # From training options
        })
        # Remove in_features if seq_len is present for TCN compatibility
        if 'seq_len' in model_options and 'in_features' in model_options:
            # seq_len takes precedence for TCN
            model_args.pop('in_features', None) # Remove in_features if present
            
    else:
        raise ValueError(f"Unknown model architecture: {architecture}")

    # # Add architecture to args AFTER specific updates to ensure it's present
    # model_args['architecture'] = architecture # Removed: Architecture is not an __init__ arg

    # --- Instantiate or Load Model --- 
    # Pop architecture before passing args, as it's not an init parameter
    model_init_args = model_args.copy() 
    model_init_args.pop('architecture', None) 

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            # Load from checkpoint, allow overriding hparams with model_init_args
            # Use strict=False if loading older checkpoints with potentially different hparams
            # Pass the original model_args (including architecture) to load_from_checkpoint
            # as it saves hyperparameters including architecture.
            net = ModelClass.load_from_checkpoint(checkpoint_path, strict=False, **model_args) 
            print(f"Model loaded successfully from {checkpoint_path}.")
            # Optionally, verify loaded hparams match current config (using model_init_args)
            # for key, val in model_init_args.items():
            #     if hasattr(net.hparams, key) and getattr(net.hparams, key) != val:
            #         print(f"  Warning: Checkpoint hparam '{key}' ({getattr(net.hparams, key)}) differs from config ({val}). Using value from config.")
        except Exception as e:
             print(f"Error loading model from checkpoint {checkpoint_path}: {e}")
             print("Attempting to initialize a new model instead.")
             net = ModelClass(**model_init_args) # Use args without architecture

    else:
        if checkpoint_path: # Checkpoint path provided but file doesn't exist
             print(f"Warning: Checkpoint file not found at {checkpoint_path}. Initializing new model.")
        print(f"Initializing new {architecture} model.")
        net = ModelClass(**model_init_args) # Use args without architecture

    return net 