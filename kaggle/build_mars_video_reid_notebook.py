"""Build the private Kaggle MARS cross-camera video-ReID demonstration."""
import base64
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle/mars_video_reid_notebook"
OUT.mkdir(exist_ok=True)
raw = subprocess.check_output(["git", "archive", "HEAD", "fastreid", "configs", "LICENSE"], cwd=ROOT)
blob = io.BytesIO()
with tarfile.open(fileobj=blob, mode="w:gz") as dest:
    with tarfile.open(fileobj=io.BytesIO(raw)) as src:
        for member in src:
            if not member.name.endswith((".so", ".pyc", ".c")):
                dest.addfile(member, src.extractfile(member) if member.isfile() else None)
    for path in [
        ROOT / "kaggle/runtime.py",
        ROOT / "kaggle/requirements-training.txt",
        ROOT / "video_demo/run_mars_reid_demo.py",
    ]:
        dest.add(path, arcname=str(path.relative_to(ROOT)))
payload = blob.getvalue()
encoded = base64.b64encode(payload).decode()
digest = hashlib.sha256(payload).hexdigest()


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


code = f'''import base64, hashlib, io, tarfile
from pathlib import Path
payload = base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
repo = Path('/tmp/secap-mars'); repo.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(payload)) as bundle:
    bundle.extractall(repo, filter='data')
ns = {{'__name__': 'secap_mars_runtime'}}
exec(compile((repo/'kaggle/runtime.py').read_text(), 'runtime.py', 'exec'), ns)
ns['REPO'] = repo
ns['ART'] = Path('/kaggle/working/secap-mars-runtime'); ns['ART'].mkdir(exist_ok=True)
py = ns['install_environment']()
art = next(Path('/kaggle/input').rglob('checkpoints/CARGO/model_best.pth')).parents[2]
mars = next(Path('/kaggle/input').rglob('bbox_test'))
out = Path('/kaggle/working/secap_mars_video_reid'); out.mkdir(exist_ok=True)
ns['run']([py, 'video_demo/run_mars_reid_demo.py',
  '--data', mars.parent, '--output-dir', out,
  '--cargo-config', art/'CARGO/config.yaml', '--cargo-checkpoint', art/'checkpoints/CARGO/model_best.pth',
  '--lagper-config', art/'LAGPeR/config.yaml', '--lagper-checkpoint', art/'checkpoints/LAGPeR/model_best.pth',
  '--identities', '60'], cwd=repo)
print('SECAP_MARS_OUTPUT', out)
'''
cells = [
    cell("markdown", "# SeCap MARS ground-truth video-ReID demo\n\nThis is a labelled cross-camera person video-ReID benchmark demonstration. MARS tracklets provide the ground-truth identity and camera label. The output videos show each model's hardest selected query, its ranked gallery candidates, the true rank, correct/incorrect result, and live benchmark statistics."),
    cell("code", code),
    cell("markdown", "## Outputs\n\n`/kaggle/working/secap_mars_video_reid/` contains one annotated cross-camera video per checkpoint and `mars_reid_report.json`. The coloured crop border denotes the benchmark's supplied person bounding box; no detector is being evaluated here."),
]
for index, item in enumerate(cells):
    item["id"] = f"secap-mars-{index}"
notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
(OUT / "secap_mars_video_reid.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
metadata = {
    "id": "rizwanabdullah05/secap-mars-video-reid-demo",
    "title": "SeCap MARS Ground-Truth Video ReID",
    "code_file": "secap_mars_video_reid.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": ["rizwanabdullah05/secap-reproduction-artifacts", "tusharbadlani0/mars-reid-data"],
    "competition_sources": [], "kernel_sources": [], "model_sources": [],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("Built", OUT)
