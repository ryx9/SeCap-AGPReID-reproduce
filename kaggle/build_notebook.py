"""Build the self-contained Kaggle notebook from the pinned author source and adapters."""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'kaggle/notebook'
OUT.mkdir(exist_ok=True)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
raw = subprocess.check_output(['git', 'archive', commit, 'fastreid', 'configs', 'tools', 'LICENSE', 'INSTALL.md', 'README.md'], cwd=ROOT)
blob = io.BytesIO()
with tarfile.open(fileobj=blob, mode='w:gz') as dest:
    with tarfile.open(fileobj=io.BytesIO(raw)) as src:
        for member in src:
            if member.name.endswith(('.so', '.pyc', '.c')):
                continue  # rebuild compiled evaluation extensions for the pinned Python
            dest.addfile(member, src.extractfile(member) if member.isfile() else None)
    for name in ['train_wrapper.py', 'runtime.py', 'requirements-training.txt', 'check_checkpoint.py', 'g2a_adapter.py', 'check_g2a.py']:
        dest.add(ROOT / 'kaggle' / name, arcname='kaggle/' + name)
bundle = blob.getvalue()
encoded = base64.b64encode(bundle).decode()
sha = hashlib.sha256(bundle).hexdigest()


def cell(kind, source):
    obj = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
    if kind == 'code':
        obj.update(execution_count=None, outputs=[])
    return obj

cells = [cell('markdown', f'''# SeCap reproduction: private datasets, GPU T4 ×2

Author source: `ryx9/SeCap-AGPReID-reproduce@{commit}`. Source is embedded for reproducibility; no GitHub push is required. Enable Internet and **GPU T4 ×2**, attach your private dataset, then **Run All**. Setup uses an isolated Python 3.10 / PyTorch 2.2.2 CUDA 12.1 environment managed by uv; no manual Conda steps.

This run is intentionally scoped to **CARGO only**. The uploaded G2A-VReID is a different benchmark from the paper's G2APS-ReID and is not run here; LAGPeR is likewise excluded so the remaining GPU budget is devoted to CARGO.

Defaults follow the paper's batch 64 / stride 16 / LR 0.008 / 120 epochs. The released YAML differs (batch 128; CARGO and G2APS-ReID stride 12). The author's multi-GPU code mines triplets and computes BatchNorm within each GPU batch; with two GPUs, these operations see 32 images each, so exact numerical equivalence to the paper's single RTX 3090 run is not promised. Set batch 128 and the desired stride below to follow the released YAML instead. All overrides are saved.

Every epoch saves metrics and a full resumable checkpoint. Best means the author's first test protocol metric `(Rank-1 + mAP)/2`; evaluation follows the released interval of five epochs. The latest two epoch files, best, last, optimizer, schedulers, AMP scaler and per-rank RNG states are preserved. Old epoch checkpoints are pruned to fit Kaggle storage; **every epoch's results remain**. Resuming restores training state at an epoch boundary; worker/sampler prefetch means it is not bitwise identical to an uninterrupted run.

Private artifact dataset snapshots are uploaded hourly and at exit. Lightweight log/progress heartbeats are uploaded every 15 minutes for external monitoring. A hardlink-aware `checkpoints.tar` avoids storing duplicate copies of the best/last/epoch aliases; it is extracted automatically on resume. The notebook stops at an epoch boundary within a quota-aware budget (at most eight hours, with one GPU hour reserved for archiving), leaving time before Kaggle's session limit to archive. Run All again to restore the latest private archive and continue. Kaggle GPU quotas can require waiting before the next session. A previously attached archive dataset is also supported.
'''), cell('code', '''SETTINGS = {
    "datasets": ["CARGO"],
    "artifact_dataset": "rizwanabdullah05/secap-reproduction-artifacts",
    "monitor_dataset": "rizwanabdullah05/secap-training-heartbeat",
    "batch_size": 64,
    "stride": [16, 16],
    "max_epoch": 90,
    "eval_period": 5,
    "session_hours": 8,
    "archive_minutes": 60,
}
'''), cell('markdown', '## Restore the exact source and run setup/training\nAll image reads, environment installation, training, intermediate checkpoints and archive uploads happen on Kaggle. No API token is included in this notebook. Kagglehub uses the notebook account authentication.'), cell('code', f'''import base64, hashlib, io, tarfile
from pathlib import Path
SOURCE_B64 = {encoded!r}
source_bytes = base64.b64decode(SOURCE_B64)
assert hashlib.sha256(source_bytes).hexdigest() == {sha!r}
Path('/tmp/secap-source.tar.gz').write_bytes(source_bytes)
repo = Path('/tmp/secap-source')
repo.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(source_bytes)) as bundle:
    bundle.extractall(repo, filter='data')
namespace = {{'__name__': 'secap_kaggle_runtime'}}
exec(compile((repo/'kaggle/runtime.py').read_text(), 'kaggle/runtime.py', 'exec'), namespace)
SETTINGS['source_commit'] = {commit!r}
namespace['main'](SETTINGS)
'''), cell('markdown', '''## Outputs and continuation
`/kaggle/working/secap-artifacts/<dataset>/` contains `epochs.jsonl`, the author's `metrics.json`, `config.yaml`, `progress.json`, `console.log`, plots, `model_best.pth`, `model_last.pth`, and the most recent epoch files. `model_final.pth` is written only at epoch 120. Exact source, environment freeze, pretrained weight hash and validation reports are included.

The private archive is `rizwanabdullah05/secap-reproduction-artifacts`. Re-run this notebook to continue from `model_last.pth`. Existing completed datasets are skipped. Keep the same batch and stride when resuming; use a different artifact dataset slug for an independent experiment. Final test results are recorded in the evaluation logs and epoch metrics. Best and final checkpoint evaluations can be run through the original `tools/train_net.py --eval-only` entry point using the saved configuration.
''')]
notebook = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.11'}}, cells=cells)
for i,c in enumerate(cells):
    c['id'] = f'secap-{i}'
(OUT / 'secap_reproduction.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')
metadata = dict(id='rizwanabdullah05/secap-reproduction-t4x2', title='SeCap Reproduction T4x2', code_file='secap_reproduction.ipynb', language='python', kernel_type='notebook', is_private=True, enable_gpu=True, enable_internet=True, machine_shape='NvidiaTeslaT4', dataset_sources=['rizwanabdullah05/person-reid-cargo-g2a-lagper', 'rizwanabdullah05/secap-reproduction-artifacts'], competition_sources=[], kernel_sources=[], model_sources=[])
(OUT / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(f'Built {OUT}/secap_reproduction.ipynb; source {len(bundle.getvalue()) if hasattr(bundle,"getvalue") else len(source_bytes) if "source_bytes" in globals() else len(blob.getvalue())} bytes; author commit {commit}')
