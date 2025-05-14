import argparse
import yaml
import sys
import os
import torch # Added for torch.compile
from importlib import import_module
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def main(cfg_path):
    torch.set_float32_matmul_precision('medium') # Set precision for float32 matrix multiplication
    # Add project root to sys.path to allow for absolute imports
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Instantiate DataModule
    # Example: dataloader: "prometheus" -> PrometheusDataModule
    dm_module_name = cfg['dataloader']
    dm_class_name_parts = [part.capitalize() for part in dm_module_name.split('_')]
    dm_class_name = "".join(dm_class_name_parts) + "DataModule"
    
    try:
        dm_module = import_module(f"dataloaders.{dm_module_name}")
        dm_class = getattr(dm_module, dm_class_name)
    except AttributeError:
        # Fallback for simple names like "prometheus" -> "PrometheusDataModule"
        # This handles cases where the template.md example (e.g. cfg['dataloader'].title()) might be too simple
        dm_class_name_simple_title = cfg['dataloader'].title().replace('_', '') + "DataModule"
        dm_class = getattr(dm_module, dm_class_name_simple_title)
        
    dm = dm_class(cfg)

    # Instantiate Model
    # Example: model_name: "om2vec" -> Om2vecModel
    model_module_name = cfg['model_name']
    model_class_name_parts = [part.capitalize() for part in model_module_name.split('_')]
    model_class_name = "".join(model_class_name_parts) # Model class name might not have "Model" suffix by convention
    
    # A common convention is PascalCase for the model class, e.g. "om2vec" -> "Om2vec" or "Om2vecModel"
    # The template suggests: getattr(model_module, f"{cfg['model_name'].title()}")
    # Let's try a more flexible approach, first assuming direct PascalCase, then adding "Model"
    
    model_module_path = f"models.{model_module_name}"
    model_module = import_module(model_module_path)
    
    potential_model_class_names = [
        model_class_name, # e.g. Om2vec
        model_class_name + "Model" # e.g. Om2vecModel
    ]
    
    model_class = None
    for name_attempt in potential_model_class_names:
        if hasattr(model_module, name_attempt):
            model_class = getattr(model_module, name_attempt)
            break
    
    if model_class is None:
        raise ImportError(f"Could not find model class in {model_module_path} from names: {potential_model_class_names}")

    if cfg.get('checkpoint') and not cfg.get('resume_training', False): # Load for testing or inference
        model = model_class.load_from_checkpoint(cfg['checkpoint'], cfg=cfg) # Pass cfg for hparams
    else: # Training or resuming training
        model = model_class(cfg)

    if cfg['training']:
        if cfg.get('logger', 'wandb') == 'csv': # Default to wandb if 'logger' not specified
            logger = CSVLogger(
                save_dir=cfg['project_save_dir'],
                name=cfg['project_name'],
                version=cfg.get('version', None)
            )
        else:
            logger = WandbLogger(
                project=cfg['project_name'],
                save_dir=cfg['project_save_dir'],
                config=cfg # Log complete config
            )
        
        callbacks = []
        if 'save_epochs' in cfg['training_options']:
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(cfg['project_save_dir'], cfg['project_name']),
                filename='{epoch}-{val_loss:.2f}',
                save_top_k=3,
                monitor='val_loss',
                mode='min',
                every_n_epochs=cfg['training_options']['save_epochs'],
            )
            callbacks.append(checkpoint_callback)
        
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)

        trainer = pl.Trainer(
            accelerator=cfg.get('accelerator', 'gpu'),
            devices=cfg.get('num_devices', 1),
            precision=cfg['training_options'].get('precision', '32-true'),
            max_epochs=cfg['training_options']['epochs'],
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=1,
        )
        
        ckpt_path_resume = cfg.get('checkpoint') if cfg.get('resume_training', False) else None
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path_resume)
        
        # Test after training if validation data is available
        if hasattr(dm, 'val_dataloader') and dm.val_dataloader() is not None:
             # Use a separate trainer for testing to ensure correct precision and no gradient calculations
            test_trainer_cfg = {
                'accelerator': cfg.get('accelerator', 'gpu'),
                'devices': cfg.get('num_devices', 1),
                'precision': cfg['training_options'].get('test_precision', cfg['training_options'].get('precision', '32-true')),
                'logger': logger # Continue logging to the same WandB run
            }
            test_trainer = pl.Trainer(**test_trainer_cfg)
            test_trainer.test(model, datamodule=dm)

    else: # Test mode
        # If a checkpoint is provided for testing, it's loaded above.
        # If no checkpoint, model is initialized from scratch (less common for test mode).
        if not cfg.get('checkpoint'):
            print("Warning: Running in test mode without a checkpoint. Model is initialized from scratch.")

        # Simplified logger for testing if not training (or use existing WandB if available)
        test_logger = WandbLogger(
            project=cfg['project_name'],
            save_dir=cfg['project_save_dir'],
            config=cfg,
            name=f"test_{cfg.get('model_name')}" # Give a distinct name for test runs
        ) if not cfg['training'] else logger


        trainer = pl.Trainer(
            accelerator=cfg.get('accelerator', 'gpu'),
            devices=cfg.get('num_devices', 1),
            precision=cfg['training_options'].get('test_precision', cfg['training_options'].get('precision', '32-true')),
            logger=test_logger
        )
        trainer.test(model, datamodule=dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PyTorch Lightning project.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args.config)