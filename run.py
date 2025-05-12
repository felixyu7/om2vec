import argparse
import yaml
from importlib import import_module
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import sys
import os
import torch

from models.om2vec_model import Om2vecModel

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser(description="Run om2vec model training or testing.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if cfg['dataloader'] == 'prometheus':
        from dataloaders.prometheus import PrometheusDataModule
        dm = PrometheusDataModule(cfg)
    else:
        raise ValueError(f"Unsupported dataloader: {cfg['dataloader']}")

    # Instantiate om2vec model
    model_class = Om2vecModel

    if cfg.get('checkpoint') and os.path.exists(cfg['checkpoint']):
        print(f"Loading model from checkpoint: {cfg['checkpoint']}")
        model = model_class.load_from_checkpoint(
            cfg['checkpoint'], 
            cfg=cfg # Pass full config for hyperparameter loading
        )
    else:
        if cfg.get('checkpoint') and not os.path.exists(cfg['checkpoint']):
            print(f"Warning: Checkpoint file {cfg['checkpoint']} not found. Training from scratch.")
        model = model_class(cfg)

    if cfg.get('training', False):
        # Training mode
        if cfg.get('logger', 'wandb') == 'csv':
            logger = CSVLogger(
                save_dir=cfg.get('project_save_dir', './csv_logs'),
                name=cfg.get('project_name', 'om2vec'),
                version=cfg.get('run_name', None) # Optional run name for CSV
            )
        else:
            logger = WandbLogger(
                project=cfg.get('project_name', 'om2vec'),
                save_dir=cfg.get('project_save_dir', './wandb_logs'),
                name=cfg.get('run_name', None) # Optional run name for WandB
            )

        callbacks = []
        if cfg.get('training_options', {}).get('save_epochs'):
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(cfg.get('project_save_dir', './wandb_logs'), cfg.get('project_name', 'om2vec')),
                filename='{epoch}-{val_loss:.2f}',
                save_top_k=cfg.get('training_options', {}).get('save_top_k', 3),
                monitor=cfg.get('training_options', {}).get('monitor_metric', 'val_loss'),
                mode=cfg.get('training_options', {}).get('monitor_mode', 'min'),
                every_n_epochs=cfg.get('training_options', {}).get('save_epochs', 1)
            )
            callbacks.append(checkpoint_callback)
        
        if cfg.get('training_options', {}).get('lr_schedule'):
            lr_monitor = LearningRateMonitor(logging_interval='step')
            callbacks.append(lr_monitor)

        trainer = pl.Trainer(
            accelerator=cfg.get('accelerator', 'cpu'),
            devices=cfg.get('num_devices', 1),
            precision=cfg.get('training_options', {}).get('precision', '32-true'),
            max_epochs=cfg.get('training_options', {}).get('epochs', 10),
            gradient_clip_val=cfg.get('training_options', {}).get('gradient_clip_val', 1.0),
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=1,
            # Add other trainer options from cfg as needed
        )
        
        ckpt_path = None
        if cfg.get('checkpoint') and cfg.get('resume_training', False) and os.path.exists(cfg['checkpoint']):
            ckpt_path = cfg['checkpoint']
            print(f"Resuming training from checkpoint: {ckpt_path}")
        elif cfg.get('checkpoint') and cfg.get('resume_training', False) and not os.path.exists(cfg['checkpoint']):
            print(f"Warning: resume_training is true but checkpoint {cfg['checkpoint']} not found. Starting fresh.")


        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    else:
        # Test mode
        trainer = pl.Trainer(
            accelerator=cfg.get('accelerator', 'cpu'),
            devices=cfg.get('num_devices', 1), # Or handle test devices separately
            precision=cfg.get('training_options', {}).get('test_precision', cfg.get('training_options', {}).get('precision', '32-true'))
            # Add other trainer options for testing as needed
        )
        if not cfg.get('checkpoint') or not os.path.exists(cfg['checkpoint']):
            print("Error: No checkpoint provided for testing, or checkpoint file not found.")
            print("Please specify a valid 'checkpoint' in the config file for test mode.")
            sys.exit(1)
            
        print(f"Starting testing with checkpoint: {cfg['checkpoint']}")
        trainer.test(model, datamodule=dm, ckpt_path=cfg['checkpoint'])

if __name__ == '__main__':
    main()