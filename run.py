import torch
import numpy as np
import lightning.pytorch as pl
import os
import yaml
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging

# Import the model factory
from models import get_model

# Import dataloaders (assuming prometheus is the main one)
from dataloaders.prometheus import PrometheusTimeSeriesDataModule

def initialize_args():
    import argparse
    parser =  argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config", # Use a more descriptive argument name
        dest="cfg_file",
        type=str,
        required=True,
        help="Path to the configuration file (e.g., configs/transformer.cfg)"
    )
    args = parser.parse_args()
    return args

if __name__=="__main__":

    torch.set_float32_matmul_precision('medium')
    # Consider making start_method configurable if needed
    # torch.multiprocessing.set_start_method('spawn') 

    args = initialize_args()
    config_path = args.cfg_file

    with open(config_path, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

    # --- Dataloader Initialization --- 
    # Keep dataloader selection logic as is
    if cfg.get('dataloader') == 'prometheus':
        dm = PrometheusTimeSeriesDataModule(cfg)
        dm.setup() # Call setup to prepare datasets
    else:
        print(f"Error: Unknown dataloader '{cfg.get('dataloader')}' specified in config.")
        exit(1)
    
    # Determine dataset size for model initialization (e.g., beta scheduling)
    # Ensure setup() has been called before accessing datasets
    if cfg.get('training', True): # Default to training mode
        # Handle cases where train_dataset might not exist if dm.setup() wasn't called correctly
        try:
            dataset_size = len(dm.train_dataset)
        except AttributeError:
             print("Error: Could not determine training dataset size. Did datamodule setup fail?")
             exit(1)
    else:
        try:
            # Need validation dataset size if not training? Typically not, but pass -1 or None.
            # Using validation dataset size if needed for non-training model setup
            dataset_size = len(dm.valid_dataset) 
        except AttributeError:
             print("Warning: Could not determine validation dataset size.")
             dataset_size = -1 # Or None, depending on factory needs

    # --- Model Initialization using Factory --- 
    # Pass the config dict, dataset_size, and config_path (for architecture inference)
    try:
        net = get_model(cfg, dataset_size, cfg_path=config_path)
    except ValueError as e:
        print(f"Error initializing model: {e}")
        exit(1)
    except KeyError as e:
        print(f"Error: Missing key in configuration for model initialization: {e}")
        exit(1)

    # --- Training/Testing Logic --- 
    # This part remains largely the same, using the initialized `net` and `dm`

    if cfg.get('training', True):
        # Wandb Logger setup
        wandb_save_dir = cfg.get('project_save_dir', '.') # Default save dir
        wandb_project_name = cfg.get('project_name', 'om2vec_runs') # Default project name
        os.makedirs(wandb_save_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = os.path.abspath(wandb_save_dir)
        wandb_logger = WandbLogger(project=wandb_project_name, save_dir=wandb_save_dir, log_model='all')
        wandb_logger.experiment.config.update(cfg) # Log the whole config

        # Callbacks
        lr_monitor = LearningRateMonitor(logging_interval='step')
        # Ensure checkpoint path is constructed safely
        checkpoint_dir = os.path.join(wandb_save_dir, wandb_project_name, wandb_logger.version, 'checkpoints')
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='model-{epoch:02d}-{val_loss:.2f}', # Use val_loss if available
            every_n_epochs=cfg.get('training_options', {}).get('save_epochs', 1),
            save_top_k=-1, # Save all checkpoints matching every_n_epochs
            save_on_train_epoch_end=False # Save based on validation preferred
        )
        
        callbacks = [checkpoint_callback, lr_monitor]
        # Add SWA if enabled and configured
        if cfg.get('training_options', {}).get('use_swa', False):
             swa_lr = cfg.get('training_options', {}).get('swa_lr', cfg.get('training_options', {}).get('lr', 1e-3))
             callbacks.append(StochasticWeightAveraging(swa_lrs=swa_lr))

        # Trainer Initialization
        trainer = pl.Trainer(
            accelerator=cfg.get('accelerator', 'auto'), 
            devices=cfg.get('num_devices', 'auto'),
            precision=cfg.get('precision', "bf16-mixed"), # Make precision configurable
            max_epochs=cfg.get('training_options', {}).get('epochs', 10), 
            log_every_n_steps=cfg.get('log_every_n_steps', 50), # Make configurable
            gradient_clip_val=cfg.get('training_options', {}).get('gradient_clip_val', 0.5), # Make configurable
            logger=wandb_logger, 
            callbacks=callbacks,
            num_sanity_val_steps=cfg.get('num_sanity_val_steps', 2) # Make configurable
        )

        # Start Training (handling resume)
        ckpt_path_resume = cfg.get('checkpoint') if cfg.get('resume_training') else None
        if ckpt_path_resume and not os.path.exists(ckpt_path_resume):
             print(f"Warning: Resume checkpoint specified but not found: {ckpt_path_resume}. Starting training from scratch.")
             ckpt_path_resume = None
        elif ckpt_path_resume:
             print(f"Resuming training from checkpoint: {ckpt_path_resume}")

        trainer.fit(model=net, datamodule=dm, ckpt_path=ckpt_path_resume)
    
    else: # Inference/Testing Mode
        # Setup logger for testing (optional)
        test_logger = WandbLogger(project=cfg.get('project_name', 'om2vec_tests'), 
                                save_dir=cfg.get('project_save_dir', '.'))
        test_logger.experiment.config.update(cfg)

        trainer = pl.Trainer(
            accelerator=cfg.get('accelerator', 'auto'),
            devices=cfg.get('num_devices', 'auto'), 
            precision=cfg.get('precision', "bf16-mixed"),
            logger=test_logger,
            # profiler='simple', # Optional profiler
            num_sanity_val_steps=0
        )
        
        # Ensure the model is loaded correctly for testing (handled by get_model)
        if not cfg.get('checkpoint'):
             print("Warning: Running in test mode without a specified checkpoint.")
        
        # Run testing
        trainer.test(model=net, datamodule=dm) # Use test_dataloader from dm

