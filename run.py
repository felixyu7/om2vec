import argparse
import yaml
import sys
import os
from importlib import import_module
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def main(cfg_path):
    # Add project root to sys.path to allow for internal imports
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Seed everything for reproducibility if specified
    if cfg.get('seed'):
        pl.seed_everything(cfg['seed'], workers=True)

    # --- Dataloader ---
    print(f"Loading dataloader: {cfg['dataloader']}")
    dm_module_name = cfg['dataloader']
    # Construct class name by capitalizing parts of the snake_case module name
    # e.g., icecube_parquet -> IcecubeParquetDataModule
    dm_class_name_parts = [part.capitalize() for part in dm_module_name.split('_')]
    dm_class_name = "".join(dm_class_name_parts) + "DataModule"
    
    try:
        dm_module = import_module(f"dataloaders.{dm_module_name}")
        dm_class = getattr(dm_module, dm_class_name)
    except ImportError:
        print(f"Error: Could not import dataloader module 'dataloaders.{dm_module_name}'.")
        sys.exit(1)
    except AttributeError:
        print(f"Error: Could not find class '{dm_class_name}' in 'dataloaders.{dm_module_name}'.")
        sys.exit(1)
        
    dm = dm_class(cfg)

    # --- Model ---
    print(f"Loading model: {cfg['model_name']}")
    model_module_name = cfg['model_name']
    # Construct class name, e.g., om2vec_model -> Om2vecModel
    model_class_name_parts = [part.capitalize() for part in model_module_name.split('_')]
    model_class_name = "".join(model_class_name_parts)

    try:
        model_module = import_module(f"models.{model_module_name}")
        model_class = getattr(model_module, model_class_name)
    except ImportError:
        print(f"Error: Could not import model module 'models.{model_module_name}'.")
        sys.exit(1)
    except AttributeError:
        print(f"Error: Could not find class '{model_class_name}' in 'models.{model_module_name}'.")
        sys.exit(1)

    if cfg.get('checkpoint') and os.path.exists(cfg['checkpoint']):
        print(f"Loading model from checkpoint: {cfg['checkpoint']}")
        model = model_class.load_from_checkpoint(cfg['checkpoint'], cfg=cfg) # Pass cfg for hparams
    else:
        if cfg.get('checkpoint'):
            print(f"Warning: Checkpoint '{cfg['checkpoint']}' not found. Initializing new model.")
        model = model_class(cfg)

    # --- Training ---
    if cfg.get('training', False):
        print("Setting up training...")
        logger = None
        if cfg.get('project_name') and cfg.get('project_save_dir') and cfg.get('logger') == 'wandb':
            logger = WandbLogger(
                name=cfg.get('run_name', None), # Optional run name for wandb
                project=cfg['project_name'],
                save_dir=cfg['project_save_dir'],
                config=cfg # Log all hyperparameters
            )
        else:
            logger = CSVLogger(
                save_dir=os.path.join(cfg['project_save_dir'], cfg['project_name']),
                name=cfg.get('run_name', None), # Optional run name for CSV
                version=cfg.get('version', None) # Optional version for CSV
            )

        callbacks = []
        if cfg.get('training_options', {}).get('save_epochs'):
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(cfg['project_save_dir'], cfg['project_name'], "checkpoints"),
                filename='{epoch}-{val_loss:.2f}', # Example filename
                save_top_k=cfg.get('training_options', {}).get('save_top_k', 1),
                monitor=cfg.get('training_options', {}).get('checkpoint_monitor', 'val_loss'), # Metric to monitor
                mode=cfg.get('training_options', {}).get('checkpoint_mode', 'min'), # 'min' or 'max'
                every_n_epochs=cfg.get('training_options', {}).get('save_epochs')
            )
            callbacks.append(checkpoint_callback)
        
        if cfg.get('training_options', {}).get('lr'):
            callbacks.append(LearningRateMonitor(logging_interval='step'))

        trainer_options = {
            'accelerator': cfg.get('accelerator', 'auto'),
            'devices': cfg.get('num_devices', 1),
            'precision': cfg.get('training_options', {}).get('precision', '32-true'),
            'max_epochs': cfg.get('training_options', {}).get('epochs', 100),
            'logger': logger,
            'callbacks': callbacks,
            'deterministic': cfg.get('seed') is not None # Ensure deterministic behavior if seed is set
        }
        
        # Add gradient clipping if specified
        if cfg.get('training_options', {}).get('grad_clip_val') is not None:
            trainer_options['gradient_clip_val'] = cfg['training_options']['grad_clip_val']

        trainer = pl.Trainer(**trainer_options)

        ckpt_path = None
        if cfg.get('checkpoint') and cfg.get('resume_training', False) and os.path.exists(cfg['checkpoint']):
            ckpt_path = cfg['checkpoint']
            print(f"Resuming training from checkpoint: {ckpt_path}")
        
        print("Starting training...")
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        
        # Test after training if test data is available
        if hasattr(dm, 'test_dataloader') and dm.test_dataloader() is not None:
            print("Starting testing after training...")
            trainer.test(model, datamodule=dm)

    # --- Testing (if not training) ---
    elif hasattr(dm, 'test_dataloader') and dm.test_dataloader() is not None:
        print("Setting up testing...")
        trainer_options = {
            'accelerator': cfg.get('accelerator', 'auto'),
            'devices': cfg.get('num_devices', 1),
            'precision': cfg.get('training_options', {}).get('test_precision', '32-true'),
            'logger': None # No logger for standalone testing unless specified
        }
        trainer = pl.Trainer(**trainer_options)
        print("Starting testing...")
        trainer.test(model, datamodule=dm)
    else:
        print("No training specified and no test dataloader available. Exiting.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run om2vec model training or testing.")
    parser.add_argument('--config', type=str, default='configs/om2vec.yaml',
                        help='Path to the configuration YAML file.')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)
        
    main(args.config)