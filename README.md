# om2vec

## Overview

This repo contains an implementation of om2vec as described in the paper [ARXIV HERE]. It is designed for use on Prometheus neutrino telescope events. om2vec uses deep learning to encode optical module (OM) timing information into a latent vector representation. Training scripts and pre-trained checkpoints are provided in this repo. Additionally, we provide a script that allows you to convert your Prometheus dataset files into latents, given a model checkpoint. 

## Directory Structure

```
om2vec/
├── configs/
│   └── train_prometheus.cfg
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

## Pre-trained checkpoints

Download pre-trained checkpoints here:

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



### Training

#### Pre-process Prometheus events

#### Configuration

