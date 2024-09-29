# om2vec
A neutrino telescope event encoder using transformer-based VAEs

## Overview

This repo contains an implementation of om2vec as described in the paper [ARXIV HERE]. It is designed for use on Prometheus neutrino telescope events. om2vec uses deep learning to encode optical module (OM) timing information into a latent vector representation. Training scripts and pre-trained checkpoints are provided in this repo. Additionally, we provide a script that allows you to convert your Prometheus dataset files into latents, given a model checkpoint. 

