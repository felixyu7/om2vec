import argparse
import yaml
import sys
import os
from importlib import import_module

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def main(cfg_path):
    # Load configuration
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Add project root to sys.path to allow for imports like 'from models.om2vec import Om2vec'
    # This assumes run.py is in the project root.
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Instantiate DataModule
    # Example: cfg['dataloader'] = "prometheus"
    # Needs dataloaders.prometheus to have PrometheusDataModule
    try:
        dm_module_name = f"dataloaders.{cfg['dataloader']}" # e.g., "dataloaders.prometheus"
        dm_module = import_module(dm_module_name)
        # Class name should be PascalCase, e.g., PrometheusDataModule
        # The template uses .title(), but module names are usually snake_case.
        # Let's assume cfg['dataloader'] is 'prometheus' and class is 'PrometheusDataModule'
        dm_class_name = "".join([s.capitalize() for s in cfg['dataloader'].split('_')]) + "DataModule"
        dm_class = getattr(dm_module, dm_class_name)
        dm = dm_class(cfg)
    except ImportError as e:
        print(f"Error importing dataloader module {dm_module_name}: {e}")
        print(f"Please ensure 'dataloaders/{cfg['dataloader']}.py' exists and contains '{dm_class_name}'.")
        sys.exit(1)
    except AttributeError as e:
        print(f"Error getting DataModule class {dm_class_name} from {dm_module_name}: {e}")
        sys.exit(1)


    # Instantiate Model
    # Example: cfg['model_name'] = "om2vec"
    # Needs models.om2vec to have Om2vec class
    try:
        model_module_name = f"models.{cfg['model_name']}" # e.g., "models.om2vec"
        model_module = import_module(model_module_name)
        # Class name should be PascalCase, e.g., Om2vec
        model_class_name = "".join([s.capitalize() for s in cfg['model_name'].split('_')])
        model_class = getattr(model_module, model_class_name)

        if cfg.get('checkpoint') and os.path.exists(cfg['checkpoint']):
            print(f"Loading model from checkpoint: {cfg['checkpoint']}")
            # Pass the full cfg to load_from_checkpoint if the model's __init__ expects it.
            # The Om2vec model expects the full cfg.
            model = model_class.load_from_checkpoint(cfg['checkpoint'], cfg=cfg)
        else:
            if cfg.get('checkpoint') and not os.path.exists(cfg['checkpoint']):
                print(f"Warning: Checkpoint file not found at {cfg['checkpoint']}. Training from scratch.")
            model = model_class(cfg)
    except ImportError as e:
        print(f"Error importing model module {model_module_name}: {e}")
        print(f"Please ensure 'models/{cfg['model_name']}.py' exists and contains '{model_class_name}'.")
        sys.exit(1)
    except AttributeError as e:
        print(f"Error getting Model class {model_class_name} from {model_module_name}: {e}")
        sys.exit(1)


    if cfg.get('training', True):
        # --- Training Mode ---
        if cfg['logger'] == 'csv':
            logger = CSVLogger(
                save_dir=cfg.get('project_save_dir', './csv_logs'),
                name=cfg.get('experiment_name', None), # Optional: experiment name for CSV
                version=cfg.get('version', 0) # Optional: version number for CSV
            )
        elif cfg['logger'] == 'wandb':
            logger = WandbLogger(
                project=cfg.get('project_name', 'om2vec_experiments'),
                save_dir=cfg.get('project_save_dir', './wandb_logs'),
                name=cfg.get('experiment_name', None) # Optional: experiment name for WandB
            )
        else:
            print(f"Error: Unsupported logger type '{cfg['logger']}'. Supported types are 'csv' and 'wandb'.")
            sys.exit(1)

        callbacks = []
        if cfg.get('training_options', {}).get('save_epochs', 0) > 0 :
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(cfg.get('project_save_dir', './checkpoints'), cfg.get('project_name', 'om2vec')),
                filename='{epoch}-{val_loss:.2f}',
                save_top_k=cfg.get('training_options', {}).get('save_top_k', 3),
                monitor='val_loss', # Monitor validation loss for saving best models
                mode='min',
                every_n_epochs=cfg.get('training_options', {}).get('save_epochs', 1)
            )
            callbacks.append(checkpoint_callback)
        
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)

        trainer = pl.Trainer(
            accelerator=cfg.get('accelerator', 'gpu'),
            devices=cfg.get('num_devices', 1),
            precision=cfg.get('training_options', {}).get('precision', '32-true'),
            max_epochs=cfg.get('training_options', {}).get('epochs', 100),
            logger=logger,
            callbacks=callbacks,
            # Add other trainer flags from cfg as needed, e.g., gradient_clip_val
            # val_check_interval, check_val_every_n_epoch, etc.
        )
        
        ckpt_path_resume = None
        if cfg.get('checkpoint') and cfg.get('resume_training', False) and os.path.exists(cfg['checkpoint']):
            ckpt_path_resume = cfg['checkpoint']
            print(f"Resuming training from checkpoint: {ckpt_path_resume}")
        elif cfg.get('checkpoint') and not cfg.get('resume_training', False) and os.path.exists(cfg['checkpoint']):
            print(f"Starting training with weights from checkpoint (not resuming trainer state): {cfg['checkpoint']}")
            # Model already loaded with weights if checkpoint was provided.
            # If you want to load weights but not resume trainer state, this is handled by model loading.
            pass


        print("Starting training...")
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path_resume)
        
        # Optionally, run test after training
        if cfg.get('run_test_after_train', False):
            print("Training finished. Running test set...")
            trainer.test(model, datamodule=dm) # Uses the best checkpoint by default if ModelCheckpoint was used

    else:
        # --- Testing Mode ---
        # In test mode, we typically load a specific checkpoint.
        # The model loading logic above already handles loading from cfg['checkpoint'].
        if not cfg.get('checkpoint') or not os.path.exists(cfg['checkpoint']):
            print("Error: Testing mode requires a valid checkpoint specified in the config.")
            sys.exit(1)
            
        print(f"Starting testing with model from: {cfg['checkpoint']}")
        trainer = pl.Trainer(
            accelerator=cfg.get('accelerator', 'gpu'),
            devices=cfg.get('num_devices', 1),
            precision=cfg.get('training_options', {}).get('test_precision', cfg.get('training_options', {}).get('precision', '32-true')),
            logger=False # No extensive logging for testing, or use a minimal logger
        )
        trainer.test(model, datamodule=dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run om2vec model training or testing.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    main(args.config)