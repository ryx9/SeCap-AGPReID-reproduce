# SeCap on Kaggle

Open [the private training notebook](https://www.kaggle.com/code/rizwanabdullah05/secap-reproduction-t4x2), select **GPU T4 ×2**, enable Internet, attach `rizwanabdullah05/person-reid-cargo-g2a-lagper`, and **Run All**. The notebook installs uv, creates an isolated Python 3.10 environment, installs matched PyTorch 2.2.2 / torchvision 0.17.2 CUDA 12.1 builds, compiles the evaluation extension, downloads and verifies the author's ImageNet ViT checkpoint, and discovers dataset roots. Conda installation is unnecessary.

The notebook source is [notebook/secap_reproduction.ipynb](notebook/secap_reproduction.ipynb). Settings are in its first code cell. It embeds author commit `df2d9dc75df35662e0e9afc444e14dec7ae4e6fc` plus the Kaggle adapters. No API token is embedded. Existing author files are unmodified.

## Published CARGO run

The checked-in notebook is currently scoped to **CARGO only**. The published CARGO model was trained for 60 epochs. Its plot, metrics, configuration, launch metadata, and data-integrity report are committed under [../results/reproduction/cargo](../results/reproduction/cargo/). For model checkpoints, contact [rizzwanabdullah@gmail.com](mailto:rizzwanabdullah@gmail.com).

## Supported experiments

- **CARGO (current notebook):** author's training data with overall, Aerial → Ground, Aerial → Aerial and Ground → Ground evaluation; it also records Ground → Aerial as a supplementary reverse-direction result.
- **LAGPeR (adapter retained):** author's training data and A→G, G→A, G→A+G split files and evaluator. It is not enabled in the current CARGO-only notebook.
- **G2A-VReID (adapter retained):** a separate video adaptation experiment. It is not enabled in the current notebook.

The G2A parser follows [the official VSLA-CLIP G2A loader](https://github.com/FHR-L/VSLA-CLIP/blob/master/datasets/data_manager.py).

Current CARGO defaults use global batch 64, four images per identity, stride 16, SGD at LR 0.008, cosine decay, AMP, and an evaluation interval of five epochs. The released YAML uses batch 128 and some datasets use stride 12; notebook overrides are saved explicitly. On two GPUs, the author's unchanged triplet mining and BatchNorm operate on 32 images per GPU. Checkpoint selection uses the author's first configured test protocol metric `(Rank-1 + mAP) / 2`.

## Checkpoints and results

`/kaggle/working/secap-artifacts/<experiment>/` contains:

- `epochs.jsonl`: every epoch's metrics and evaluations; new wrapper versions also include mean training losses.
- `metrics.json`, TensorBoard events, `console.log`, `config.yaml`, `launch.json`.
- `model_last.pth`: last completed epoch, full model, optimizer, schedulers, AMP scaler and per-rank RNG states.
- `model_best.pth`: best evaluated model, with full resumable state; `best-evaluation.json` records its metrics.
- Latest two `model_epoch_NNNN.pth` files. Older model files are pruned for storage; epoch results remain.
- `model_final.pth`: written after a configured training run finishes.
- `progress.json`, `summary.json`, `training.png`.

The root includes exact source, dataset validation, environment freeze, run settings and the pretrained checkpoint hash. Source images and dependency caches stay out of the archive.

A separate private `secap-training-heartbeat` dataset receives lightweight log/progress snapshots every 15 minutes. Private full snapshots go to [secap-reproduction-artifacts](https://www.kaggle.com/datasets/rizwanabdullah05/secap-reproduction-artifacts) hourly and at exit. A hardlink-aware `checkpoints.tar` stores shared checkpoint aliases once. Resume automatically extracts it on Kaggle. The notebook checks archive write access before training; it uses Kagglehub's notebook authentication, with no token in notebook code or datasets. For checkpoint access, contact [rizzwanabdullah@gmail.com](mailto:rizzwanabdullah@gmail.com).

Training stops at a completed epoch within a quota-aware budget (at most eight hours, with one GPU hour reserved for archiving) to allow archiving before Kaggle's session limit. Run All again to restore the last epoch and continue, skipping completed experiments. Keep batch size, stride and evaluation interval unchanged when resuming; choose a new archive slug for a different experiment. Restoring optimizer/scheduler/scaler/RNG state supports continuation, but sampler/worker prefetch does not permit a claim of bitwise-identical continuation.

## Launch and monitor through uv

From this repository:

```bash
python3 kaggle/build_notebook.py
uv run kaggle kernels push -p kaggle/notebook --accelerator NvidiaTeslaT4
uv run python kaggle/watch.py --loop --resume-sessions
```

Start a detached watcher with `uv run python kaggle/start_monitor.py`. The watcher checks source readiness every five minutes until ready, then status/logs every forty-five minutes. It writes only small logs/metadata to ignored `kaggle/monitor/`. Completed budget-limited sessions can start a fresh resumable session. Failed training stops automatic resubmission for inspection; transient API errors are retried on the next check. It must remain running on an awake, network-connected host. Kaggle GPU quota errors may require waiting before another session.

Live logs can be retrieved with the resolved run ID even when the standard CLI log endpoint returns an empty log. To cancel a specific version, use `uv run python kaggle/session_tools.py cancel --version NUMBER`; this resolves the version through an authenticated output redirect and leaves other versions intact.

Check immediately:

```bash
uv run python kaggle/watch.py
uv run python kaggle/session_tools.py logs --seconds 10
```

Once an experiment finishes, the watcher fetches final plots, reports, configs and the completed best/last model files into ignored `results/kaggle/`. Intermediate images/checkpoints are not downloaded locally. Full artifacts remain on Kaggle.

## Validation

The notebook schema and all cell syntax are validated locally. `python3 -m unittest kaggle/test_runtime.py -v` verifies data-root discovery, split normalization, hardlink-aware checkpoint archives, and actual archive restoration with small fixtures. GPU-side preflight scripts check checkpoint best/last selection, retention, actual optimizer/scheduler/scaler restoration, and G2A protocol parsing/frame completeness/video feature pooling before long training. These GPU-side tests require the Kaggle runtime and must pass there before training is considered verified.
