# Reproduction artifacts

This directory contains small, reviewable outputs exported from the private Kaggle artifact dataset. Raw images and large model checkpoints are intentionally not tracked in Git.

## CARGO epoch-60 run

[`cargo_epoch60/`](cargo_epoch60/) records the best/last state after the user-requested stop at epoch 60 of a planned 90-epoch CARGO run on two T4 GPUs. It contains the plotted training history, epoch metrics, best evaluation, launch configuration, and the training-image integrity audit.

The run is reproducible from the committed Kaggle notebook and source adapters. It must not be represented as a completed 90-epoch reproduction or as a reproduction of the paper's G2APS-ReID results.
