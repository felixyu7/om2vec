# om2vec

## Overview

This repo contains an implementation of om2vec as described in the paper [ARXIV HERE]. It is designed for use on Prometheus neutrino telescope events. om2vec uses deep learning to encode optical module (OM) timing information into a latent vector representation. Training scripts and pre-trained checkpoints are provided in this repo. Additionally, we provide a script that allows you to convert your [Prometheus](https://github.com/Harvard-Neutrino/prometheus) dataset files into latents, given a model checkpoint. 

## Directory Structure

```
om2vec/
├── configs/
│   └── example.cfg
├── convert_to_latents.py
├── dataloaders/
│   ├── data_utils.py
│   └── prometheus.py
├── LICENSE
├── README.md
├── requirements.txt
├── run.py
├── scripts/
│   └── preprocess_prometheus.py
├── utils.py
└── vae.py
```

## Requirements

- PyTorch
- PyTorch Lightning
- NumPy
- Awkward Arrays
- yaml

You can install the requirements by running `pip install -r requirements.txt`.

## Pre-trained checkpoints & Datasets

Download pre-trained checkpoints here: [Google Drive](https://drive.google.com/drive/folders/1xSpbDQEmTRJ45xJNLwFkJnQAjgkjs4aB?usp=sharing), for latent_dim 32, 64, 128 models respectively.

Additionally, Prometheus datasets are being uploaded to the [Harvard dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NP1D9W&faces-redirect=true).

## Usage

### Converting your Prometheus datasets into latents

You can convert your existing Prometheus Parquet data files into om2vec latents by using the script `convert_to_latents.py`. The script will take in Prometheus files and output files in the same format, with an added column `latents`. 

Provide the path to the directory containing the Prometheus datasets, the path to the output directory, and the path to the model checkpoint (you can train your own or use the provided pre-trained checkpoints above), like so

`python convert_to_latents.py --input_dir /PATH_TO_PROMETHEUS_DATASETS/ --output_dir /PATH_TO_OUTPUT_DIR/ --ckpt /PATH_TO_CHECKPOINT/`

Additional optional arguments to the script are detailed below:
- `chunk_size`: number of events per Parquet file in the output (default: 5000)
- `max_time`: maximum time (in ns) when binning the hits from Prometheus (default: 6400)
- `num_bins`: number of bins to use when binning the hits from Prometheus (default: 6400)

**IMPORTANT NOTE**: `max_time` and `num_bins` should match with what the model checkpoint was trained with (pre-trained checkpoints are trained with the default values of 6400)

The output Parquet files generated by the script contains an extra column called `latents` which contains the latent representations from om2vec.

```
[In]: import awkward as ak
 ...: data = ak.from_parquet('./example_output.parquet')
 ...: data.fields

[Out]: ['mc_truth', 'photons', 'latents']

[In]: len(data)

[Out]: 5000

[In]: # list of OMs contained in the first event
 ...: data[0].latents

[Out]: <Array [[49, 1, -0.373, ..., -1.66, 0.116], ...] type='8 * var * float64'>

[In]: # first OM in the first event
 ...: data[0].latents[0]

[Out]: <Array [49, 1, -0.373, 1.1, ..., 0.928, -1.66, 0.116] type='66 * float64'>
```

Each latent representation of an OM contains [string_id, sensor_id, latents...]. In the above example we use the 64-dim om2vec latent so each contains 66 values in total. 

### Training

Use `run.py` and a configuration file to run training and/or inference.

`python run.py -c ./configs/example.cfg`

#### Pre-process Prometheus events

Before training, you will need to pre-process events from raw Prometheus output. See `scripts/preprocess_prometheus.py`. This script will generate Parquet files of individual binned OM timing distributions from Prometheus events. You need to provide an `input_dir` containing the Prometheus parquet files and an `output_dir`.

`python scripts/preprocess_prometheus.py --input_dir /PATH_TO_PROMETHEUS_DATASETS/ --output_dir /PATH_TO_OUTPUT_DIR/`

You can also provide the following optional arguments:
- `chunk_size`: Number of individual binned OM timing distributions per file (Default: 250000)
- `max_time`: maximum time (in ns) when binning the hits from Prometheus (default: 6400)
- `num_bins`: number of bins to use when binning the hits from Prometheus (default: 6400)

#### Configuration

We provide an example configuration file in `configs/example.cfg`. See details on the options below:
- `accelerator`: type of PyTorch Lightning accelerator to use (e.g. 'gpu')
- `num_devices`: number of devices to deploy model on
- `training`: boolean for training mode. Use `False` to run in test/inference mode.
- `dataloader`: type of dataloader to use. Implemented dataloaders: 'prometheus'
- `checkpoint`: path to checkpoint file if loading model from checkpoint
- `resume_training`: Set to `True` if loading from a checkpoint and resuming training
- `project_name`: wandb project name
- `project_save_dir`: wandb project save directory
- `model_options`: arguments to be passed into the om2vec network initialization
- `data_options.train_data_files`: list of paths to directories containing the training data (pre-processed)
- `data_options.train_data_file_ranges`: list of ranges ([x, y]) indicating which/how many files in each directory to use. must be same length as `train_data_files`
- `data_options.valid_data_files`: list of paths to directories containing the validation data (pre-processed)
- `data_options.valid_data_file_ranges`: list of ranges ([x, y]) indicating which/how many files in each directory to use. must be same length as `valid_data_files`
- `data_options.file_chunk_size`: number of timing distributions per file. set to same as `chunk_size` during pre-processing
- `training_options`: various training hyperparameters
