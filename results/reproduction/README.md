# Reproduction artifacts

This directory contains small, reviewable outputs exported from the private Kaggle artifact dataset. Raw images and large model checkpoints are intentionally not tracked in Git.

## Published runs

[`cargo/`](cargo/) records the CARGO model trained for 60 epochs on two T4 GPUs. It contains the plotted training history, metrics, launch configuration, and the training-image integrity audit.

[`lagper/`](lagper/) records the LAGPeR training history and evaluation outputs.

The run materials are reproducible from the committed Kaggle notebook and source adapters. For checkpoint access, contact [rizzwanabdullah@gmail.com](mailto:rizzwanabdullah@gmail.com).
