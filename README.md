# om2vec

Generative VAE-Transformer-CNF for Photon Pulse Series Modeling.

## Project Overview

This project implements a deep generative model for simulating photon pulse series, particularly relevant for neutrino telescope data analysis. The architecture combines:

1.  A **Variational Autoencoder (VAE)** framework with a hybrid latent space.
2.  A **Bidirectional Transformer Encoder** for feature extraction from photon sequences.
3.  A **Conditional Normalizing Flow (CNF) Decoder** (using `zuko`'s Neural Spline Flows) to model the joint probability of photon arrival times and charges.

The model is designed to process data on a per-sensor basis, learning to generate realistic pulse series conditioned on sensor-specific information and summary event characteristics.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd om2vec
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt 
    # (Note: A requirements.txt file will need to be created)
    # Key dependencies will include: torch, pytorch-lightning, pyyaml, wandb, awkward-array, zuko
    ```

## Configuration

All training, data, and model parameters are managed through a YAML configuration file.

*   Create a configuration file (e.g., `configs/om2vec_config.yaml`) based on the schema in `template.md` and `plan.md`.
*   Key sections include:
    *   `project_name`, `project_save_dir`
    *   `training` (boolean flag)
    *   `accelerator`, `num_devices`
    *   `dataloader` (e.g., "prometheus")
    *   `model_name` (e.g., "om2vec_model")
    *   `data_options`: paths to data, sequence length, etc.
    *   `model_options`: dimensions for latent space, Transformer, CNF.
    *   `training_options`: batch size, learning rate, epochs, precision.

## Running the Model

### Training

To train the model, run:
```bash
python run.py --config configs/your_config_file.yaml
```
Ensure `training: true` is set in your configuration file. Training logs and checkpoints will be saved to the directory specified by `project_save_dir` and logged to Weights & Biases if configured.

### Testing

To test a trained model, ensure you have a checkpoint file. Set `training: false` in your configuration file and provide the path to the checkpoint:
```bash
python run.py --config configs/your_config_file.yaml
```
The `checkpoint` field in the config must point to a valid `.ckpt` file.

## Dependencies
* PyTorch
* PyTorch Lightning
* YAML (pyyaml)
* Weights & Biases (wandb)
* Awkward Array
* Zuko

(A `requirements.txt` should be generated based on the final imports used.)