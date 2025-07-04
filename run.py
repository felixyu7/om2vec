import torch
import numpy as np
import lightning.pytorch as pl
import os
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from vae import NT_VAE

import yaml

def initialize_args():
    import argparse
    parser =  argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        dest="cfg_file",
        type=str,
        required=True
    )
    args = parser.parse_args()
    return args

if __name__=="__main__":

    torch.set_float32_matmul_precision('medium')
    torch.multiprocessing.set_start_method('spawn')

    args = initialize_args()

    with open(args.cfg_file, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

    # initialize dataloaders
    if cfg['dataloader'] == 'prometheus':
        from dataloaders.prometheus import PrometheusTimeSeriesDataModule
        dm = PrometheusTimeSeriesDataModule(cfg)
    else:
        print("Unknown dataloader!")
        exit()
    dm.setup()
    
    if cfg['training']:
        dataset_size = len(dm.train_dataset)
    else:
        dataset_size = len(dm.valid_dataset)
    
    if cfg['checkpoint'] != '':
        # initialize models
        print("Loading checkpoint: ", cfg['checkpoint'])
        net = NT_VAE.load_from_checkpoint(cfg['checkpoint'])
    else:
        # Pass model_options and relevant data_options directly
        # The VAE's __init__ will use self.hparams to access these
        model_hparams = {**cfg['model_options'], **cfg['training_options']}
        # Add max_seq_len_padding from data_options as it's needed by the model
        model_hparams['max_seq_len_padding'] = cfg['data_options']['max_seq_len_padding']
        # The dataset_size for beta annealing will be handled differently or passed via datamodule
        # For now, remove direct dataset_size/batch_size from VAE constructor if they are for beta annealing steps
        
        # Consolidate all necessary hparams for NT_VAE
        # NT_VAE will access them via self.hparams
        vae_init_args = {
            "latent_dim": cfg['model_options']['latent_dim'],
            "embed_dim": cfg['model_options']['embed_dim'],
            "alpha": cfg['model_options']['alpha'],
            "lambda_": cfg['model_options']['lambda_'],
            "max_seq_len_padding": cfg['data_options']['max_seq_len_padding'],
            "transformer_encoder_layers": cfg['model_options']['transformer_encoder_layers'],
            "transformer_encoder_heads": cfg['model_options']['transformer_encoder_heads'],
            "transformer_encoder_ff_dim": cfg['model_options']['transformer_encoder_ff_dim'],
            "transformer_encoder_dropout": cfg['model_options']['transformer_encoder_dropout'],
            "transformer_decoder_layers": cfg['model_options']['transformer_decoder_layers'],
            "transformer_decoder_heads": cfg['model_options']['transformer_decoder_heads'],
            "transformer_decoder_ff_dim": cfg['model_options']['transformer_decoder_ff_dim'],
            "transformer_decoder_dropout": cfg['model_options']['transformer_decoder_dropout'],
            "batch_size": cfg['training_options']['batch_size'],
            "lr": cfg['training_options']['lr'],
            "lr_schedule": cfg['training_options']['lr_schedule'],
            "weight_decay": cfg['training_options']['weight_decay'],
            "charge_wasserstein_loss_weight": cfg['model_options']['charge_wasserstein_loss_weight'],
            "interval_wasserstein_loss_weight": cfg['model_options']['interval_wasserstein_loss_weight'],
        }

        net = NT_VAE(**vae_init_args)
    
    if cfg['training']:
        if cfg['logger'] == 'wandb':
            # initialise the wandb logger and name your wandb project
            os.environ["WANDB_DIR"] = os.path.abspath(cfg['project_save_dir'])
            wandb_logger = WandbLogger(project=cfg['project_name'], save_dir=cfg['project_save_dir'], log_model='all')

            # add your batch size to the wandb config
            wandb_logger.experiment.config["batch_size"] = cfg['training_options']['batch_size']
        else:
            wandb_logger = CSVLogger(cfg['project_save_dir'], name=cfg['project_name'])

        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(dirpath=cfg['project_save_dir'] + '/' + cfg['project_name'] + '/checkpoints',
                                              filename='model-{epoch:02d}-{val_loss:.2f}.ckpt', 
                                              every_n_epochs=cfg['training_options']['save_epochs'],
                                              save_on_train_epoch_end=True)
        trainer = pl.Trainer(accelerator=cfg['accelerator'], 
                             devices=cfg['num_devices'],
                             precision=cfg['training_options']['precision'],
                             max_epochs=cfg['training_options']['epochs'], 
                             log_every_n_steps=1, 
                            #  overfit_batches=10,
                             gradient_clip_val=cfg['training_options']['gradient_clip_val'],
                             logger=wandb_logger, 
                             callbacks=[checkpoint_callback, lr_monitor],
                             num_sanity_val_steps=2)
        if cfg['resume_training']:
            print("Resuming training from checkpoint ", cfg['checkpoint'])
            trainer.fit(model=net, datamodule=dm, ckpt_path=cfg['checkpoint'])
        else:
            trainer.fit(model=net, datamodule=dm)
    else:
        logger = WandbLogger(project=cfg['project_name'], save_dir=cfg['project_save_dir'])
        logger.experiment.config["batch_size"] = cfg['training_options']['batch_size']
        trainer = pl.Trainer(accelerator=cfg['accelerator'], 
                             precision=cfg['training_options']['test_precision'],
                             profiler='simple', 
                             logger=logger,
                             num_sanity_val_steps=0)
        trainer.test(model=net, datamodule=dm)

