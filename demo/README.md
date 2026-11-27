# SeCap CARGO web demo

This folder contains a local browser-based ReID workbench and a separate reproduced-results page.

The workbench has three modes:

- **Gallery retrieval:** enter a query image/directory and gallery directory; it ranks the ten closest gallery images for every query.
- **Video annotation:** enter a local video path and gallery directory; it writes an annotated MP4 with person boxes, nearest gallery label, similarity, and live FPS.
- **Live multi-camera ReID:** provide one webcam index, video path, or RTSP URL per line and view the annotated streams in the browser.

It uses the exact `CARGO/model_best.pth` checkpoint supplied in `training_artifacts.zip` (best evaluation at epoch 60; overall Rank-1 61.54 and mAP 54.26). The checkpoint is deliberately not versioned in Git; extract it from the supplied archive instead:

```bash
python demo/setup_artifacts.py
```

Install and run through the project UV environment:

```bash
uv sync
uv run python demo/app.py --host 0.0.0.0 --port 8000
```

For a non-default checkpoint or detector, use `--config`, `--checkpoint`, and
`--detector`. The detector weights are downloaded by Ultralytics on first use.
All paths and output videos stay on the local machine (`demo/outputs/`).

The runtime dependencies are maintained in the project manifest with `uv add torch numpy pillow yacs pyyaml termcolor`. Open `http://localhost:8000`. The model loads only on the first comparison. GPU is used when PyTorch detects CUDA; otherwise the same checkpoint runs on CPU, with slower first inference.

The **Reproduced results** page records the supplied CARGO and LAGPeR best-checkpoint evaluations. It explicitly avoids treating the two datasets as a common leaderboard.

---

# Legacy FastReID CLI demo

We provide a command line tool to run a simple demo of builtin models.

You can run this command to get cosine similarites between different images

```bash
CUDA_VISIBLE_DEVICES=5 python demo/visualize_result.py --config-file logs/dukemtmc/mgn_R50-ibn/config.yaml \
--parallel --vis-label --dataset-name DukeMTMC --output logs/mgn_duke_vis \
--opts MODEL.WEIGHTS logs/dukemtmc/mgn_R50-ibn/model_final.pth


CUDA_VISIBLE_DEVICES=4 python demo/visualize_result.py --config-file logs/GAReID/Vit_base_new/config.yaml --parallel --vis-label --dataset-name GAReID_G2A --output VDT-AGPRID/logs/GAReID/G2A_baseline --opts MODEL.WEIGHTS logs/GAReID/Vit_base_new/model_final.pth

CUDA_VISIBLE_DEVICES=5 python demo/visualize_result.py --config-file logs/GAReID/VQD_128/config.yaml --parallel --vis-label --dataset-name GAReID_G2A --output VDT-AGPRID/logs/GAReID/G2A_ours --opts MODEL.WEIGHTS logs/GAReID/VQD_128/model_final.pth

CUDA_VISIBLE_DEVICES=4 python demo/visualize_result.py --config-file logs/GAReID/Vit_base_new/config.yaml --parallel --vis-label --dataset-name GAReID_A2GG --output VDT-AGPRID/logs/GAReID/A2GG_baseline --opts MODEL.WEIGHTS logs/GAReID/Vit_base_new/model_final.pth

CUDA_VISIBLE_DEVICES=5 python demo/visualize_result.py --config-file logs/GAReID/VQD_128/config.yaml --parallel --vis-label --dataset-name GAReID_A2GG --output VDT-AGPRID/logs/GAReID/A2GG_ours --opts MODEL.WEIGHTS logs/GAReID/VQD_128/model_final.pth

CUDA_VISIBLE_DEVICES=4 python demo/visualize_result.py --config-file logs/GAReID/Vit_base_new/config.yaml --parallel --vis-label --dataset-name GAReID_G2AA --output VDT-AGPRID/logs/GAReID/G2AA_baseline --opts MODEL.WEIGHTS logs/GAReID/Vit_base_new/model_final.pth

CUDA_VISIBLE_DEVICES=5 python demo/visualize_result.py --config-file logs/GAReID/VQD_128/config.yaml --parallel --vis-label --dataset-name GAReID_G2AA --output VDT-AGPRID/logs/GAReID/G2AA_ours --opts MODEL.WEIGHTS logs/GAReID/VQD_128/model_final.pth

CUDA_VISIBLE_DEVICES=6 python demo/visualize_result.py --config-file logs/GAReID/Vit_base_new/config.yaml --parallel --vis-label --dataset-name GAReID_G2AG --output VDT-AGPRID/logs/GAReID/G2AG_baseline --opts MODEL.WEIGHTS logs/GAReID/Vit_base_new/model_final.pth

CUDA_VISIBLE_DEVICES=6 python demo/visualize_result.py --config-file logs/GAReID/VQD_128/config.yaml --parallel --vis-label --dataset-name GAReID_G2AG --output VDT-AGPRID/logs/GAReID/G2AG_ours --opts MODEL.WEIGHTS logs/GAReID/VQD_128/model_final.pth


CUDA_VISIBLE_DEVICES=6 python demo/visualize_result.py --config-file logs/AGReID/sbs_vit_base/config.yaml --parallel --vis-label --dataset-name AG_ReID --output VDT-AGPRID/logs/AG_ReID/AG_baseline --opts MODEL.WEIGHTS logs/AGReID/sbs_vit_base/model_final.pth

CUDA_VISIBLE_DEVICES=5 python demo/visualize_result.py --config-file logs/AGReID/VQD_128_10view/config.yaml --parallel --vis-label --dataset-name AG_ReID --output VDT-AGPRID/logs/AG_ReID/AG_ours --opts MODEL.WEIGHTS logs/AGReID/VQD_128_10view/model_final.pth

CUDA_VISIBLE_DEVICES=4 python demo/visualize_result.py --config-file logs/AGReID/VQD_128_10view/config.yaml --parallel --vis-label --dataset-name AG_ReID_G2A --output VDT-AGPRID/logs/AGReID/GA_ours --opts MODEL.WEIGHTS logs/AGReID/VQD_128_10view/model_final.pth

CUDA_VISIBLE_DEVICES=6 python demo/visualize_result.py --config-file logs/AGReID/sbs_vit_base/config.yaml --parallel --vis-label --dataset-name AG_ReID_G2A --output VDT-AGPRID/logs/AGReID/GA_baseline --opts MODEL.WEIGHTS logs/AGReID/sbs_vit_base/model_final.pth


CUDA_VISIBLE_DEVICES=0 python demo/visualize_result.py --config-file logs/CARGO/secap_64/config.yaml --parallel --vis-label --dataset-name CARGO_AA --output VDT-AGPRID/logs/CARGO/secap_64 --opts MODEL.WEIGHTS logs/CARGO/secap_64/model_best.pth

CUDA_VISIBLE_DEVICES=0 python demo/visualize_tSNE.py --config-file logs/GAReID/VQD_128/config.yaml --vis-label --dataset-name GAReID --output VDT-AGPRID/logs/GAReID/tSNE --opts MODEL.WEIGHTS logs/GAReID/VQD_128/model_best.pth DATALOADER.NUM_INSTANCE 32 SOLVER.IMS_PER_BATCH 64

CUDA_VISIBLE_DEVICES=1 python demo/visualize_tSNE.py --config-file logs/GAReID/Vit_base/config.yaml --vis-label --dataset-name GAReID --output VDT-AGPRID/logs/GAReID/tSNE/vit --opts MODEL.WEIGHTS logs/GAReID/Vit_base/model_best.pth DATALOADER.NUM_INSTANCE 32 SOLVER.IMS_PER_BATCH 64

```
