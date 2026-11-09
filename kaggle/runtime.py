"""Runs inside Kaggle's notebook Python; heavy data and environments stay on Kaggle."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request

WORK = Path('/kaggle/working')
ART = WORK / 'secap-artifacts'
REPO = Path('/tmp/secap-source')
ENV = Path('/tmp/secap-env')
DATA = Path('/tmp/secap-datasets')
STARTED = time.time()


def run(cmd, **kwargs):
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kwargs)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def prepare_data():
    print('SECAP_SETUP: discovering dataset roots without scanning image trees', flush=True)
    DATA.mkdir(exist_ok=True)
    report = {}
    roots = {'LAGPeR': [], 'CARGO': [], 'G2AVReID': []}
    # Discover container directories without descending through hundreds of
    # thousands of image files on the mounted dataset filesystem.
    image_dirs = {'bounding_box_train', 'bounding_box_test_aerial', 'bounding_box_test_ground',
                  'query_aerial', 'query_ground', 'bounding_box_gallery', 'bbox_train', 'bbox_test'}
    for current, dirs, files in os.walk('/kaggle/input'):
        found = None
        if 'bounding_box_train' in dirs and 'exp5_G2A+G.txt' in files:
            found = 'LAGPeR'
        elif {'train', 'query', 'gallery'} <= set(dirs):
            found = 'CARGO'
        elif {'bbox_train', 'bbox_test', 'info'} <= set(dirs):
            found = 'G2AVReID'
        if found:
            print('SECAP_DATASET_ROOT', found, current, flush=True)
            roots[found].append(Path(current))
            dirs[:] = []
        else:
            dirs[:] = [d for d in dirs if d not in image_dirs]
    for name in ['LAGPeR', 'CARGO']:
        matches = roots[name]
        if len(matches) != 1:
            raise RuntimeError(f'{name}: expected one dataset root, found {matches}. Check the private input attachment.')
        source = matches[0]
        target = DATA / name
        if not target.exists():
            target.mkdir()
            for item in source.iterdir():
                if item.suffix == '.txt':
                    # Normalize line endings/final newline only; preserve author split membership.
                    lines = item.read_text().splitlines()
                    (target / item.name).write_text('\n'.join(s.replace('\\', '/') for s in lines) + '\n')
                else:
                    (target / item.name).symlink_to(item)
        train_path = target / ('bounding_box_train' if name == 'LAGPeR' else 'train')
        print('SECAP_VALIDATE_TRAIN', name, flush=True)
        train_files = list(train_path.glob('*/*.jpg'))
        if not train_files:
            raise RuntimeError(f'No training images at {train_path}')
        report[name] = dict(source=str(source), train_images=len(train_files))
        if name == 'LAGPeR':
            entries = {}
            def mounted_file_exists(relative):
                path = target / relative
                if path.parent not in entries:
                    try:
                        entries[path.parent] = set(os.listdir(path.parent))
                    except FileNotFoundError:
                        entries[path.parent] = set()
                return path.name in entries[path.parent]
            for split in ['exp3_A2G+.txt', 'exp4_G2A+.txt', 'exp5_G2A+G.txt']:
                lines = (target / split).read_text().splitlines()
                print('SECAP_VALIDATE_SPLIT', split, len(lines), flush=True)
                missing = [s for s in lines if not mounted_file_exists(s)]
                if missing:
                    raise RuntimeError(f'{split}: {len(missing)} referenced files missing; first={missing[:3]}')
                report[name][split] = len(lines)
    g2a = roots['G2AVReID']
    if len(g2a) != 1:
        raise RuntimeError(f'Expected one G2A-VReID root, found {g2a}')
    if not (DATA / 'G2A-VReID').exists():
        (DATA / 'G2A-VReID').symlink_to(g2a[0])
    report['G2AVReID'] = dict(source=str(g2a[0]), protocol='Official tracklet queries, frame training and mean-pooled video evaluation; separate from SeCap paper G2APS-ReID.')
    write_json(ART / 'dataset-validation.json', report)
    print(json.dumps(report, indent=2), flush=True)
    os.environ['FASTREID_DATASETS'] = str(DATA)


def install_environment():
    run([sys.executable, '-m', 'pip', 'install', '--quiet', 'uv==0.13.1', 'kagglehub==1.0.3'])
    uv = shutil.which('uv')
    if not uv:
        uv = str(Path(sys.executable).parent / 'uv')
    os.environ['UV_CACHE_DIR'] = '/tmp/secap-uv-cache'
    if not (ENV / 'bin/python').exists():
        run([uv, 'venv', '--python', '3.10', ENV])
    py = ENV / 'bin/python'
    run([uv, 'pip', 'install', '--python', py, 'torch==2.2.2', 'torchvision==0.17.2',
         '--index-url', 'https://download.pytorch.org/whl/cu121'])
    run([uv, 'pip', 'install', '--python', py, '-r', REPO / 'kaggle/requirements-training.txt'])
    os.environ['PATH'] = str(ENV / 'bin') + ':' + os.environ['PATH']
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['OMP_NUM_THREADS'] = '2'
    run([py, '-c', "import torch; print(torch.__version__); print([(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]); assert torch.cuda.device_count()==2, 'Select GPU T4 x2'"])
    with (ART / 'environment-freeze.txt').open('w') as f:
        subprocess.run([uv, 'pip', 'freeze', '--python', str(py)], stdout=f, check=True)
    run([py, 'setup.py', 'build_ext', '--inplace'], cwd=REPO / 'fastreid/evaluation/rank_cylib')
    return py


def quarantine_cargo(py):
    """Validate CARGO training images once and exclude only invalid ones.

    Query/gallery are deliberately never filtered: an invalid evaluation file
    must be reported rather than silently changing the published protocol.
    """
    source = (DATA / 'CARGO' / 'train').resolve().parent
    code = r'''
import concurrent.futures, json, os, sys
from PIL import Image
from pathlib import Path
source, repo, report_path = map(Path, sys.argv[1:])
images = sorted((source / "train").glob("Cam*/*.jpg"))
def validate(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except Exception as error:
        return {"path": str(path.relative_to(source)), "error": type(error).__name__}
bad_relative = []
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    for index, result in enumerate(pool.map(validate, images), 1):
        if result:
            bad_relative.append(result)
        if index % 2000 == 0:
            print("SECAP_CARGO_SCAN", index, len(images), "invalid", len(bad_relative), flush=True)
bad_names = [Path(row["path"]).name for row in bad_relative]
cargo = repo / "fastreid/data/datasets/cargo.py"
text = cargo.read_text()
marker = "# SECAP_CARGO_QUARANTINE"
if marker not in text:
    text = text.replace("import pdb\n", "import pdb\n\n" + marker + "\n_BAD_TRAIN_FILENAMES = frozenset(filter(None, os.environ.get('SECAP_CARGO_BAD_TRAIN_IMAGES', '').split(os.pathsep)))\n")
    text = text.replace("        data = []\n", "        if is_train:\n            img_paths = [p for p in img_paths if os.path.basename(p) not in _BAD_TRAIN_FILENAMES]\n        data = []\n")
    cargo.write_text(text)
report = {"source": str(source), "validation": "parallel PIL verification of all training images", "train_images_checked": len(images), "excluded_train_images": bad_relative, "query_gallery_modified": False}
report_path.write_text(json.dumps(report, indent=2))
print("SECAP_CARGO_INTEGRITY", json.dumps(report), flush=True)
'''
    report = ART / 'CARGO-data-integrity.json'
    run([py, '-c', code, source, REPO, report])
    details = json.loads(report.read_text())
    os.environ['SECAP_CARGO_BAD_TRAIN_IMAGES'] = os.pathsep.join(
        Path(row['path']).name for row in details['excluded_train_images'])


def pretrained():
    dest = Path('/tmp/jx_vit_base_p16_224-80ecf9dd.pth')
    url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/' + dest.name
    if not dest.exists():
        print('SECAP_PRETRAIN_DOWNLOAD: fetching ImageNet ViT weights', flush=True)
        partial = dest.with_suffix('.part')
        for attempt in range(3):
            try:
                downloaded = 0
                next_report = 64 * 1024 * 1024
                with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            print('SECAP_PRETRAIN_BYTES', downloaded, flush=True)
                            next_report += 64 * 1024 * 1024
                partial.replace(dest)
                break
            except Exception:
                partial.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                print('SECAP_PRETRAIN_RETRY', attempt + 1, flush=True)
    digest = hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest()
    if not digest.startswith('80ecf9dd'):
        raise RuntimeError(f'Pretrained weight SHA256 mismatch: {digest}')
    write_json(ART / 'pretrained.json', dict(url=url, sha256=digest))
    print('SECAP_PRETRAIN_READY', digest, flush=True)
    return dest


def snapshot_upload(handle, note):
    import kagglehub
    # Epoch .pth files are immutable; aliases are atomically replaced. Hardlink
    # them into a stable staging snapshot, avoiding extra multi-GB copies.
    stage = WORK / '.secap-upload'
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    for p in ART.rglob('*'):
        if not p.is_file() or p.name.endswith('.tmp') or p.name.startswith('_'):
            continue
        dest = stage / p.relative_to(ART)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == '.pth':
            try:
                os.link(p, dest)
            except FileNotFoundError:
                pass
        else:
            shutil.copy2(p, dest)
    # Tar preserves hardlinks: epoch/last/best aliases do not multiply uploaded
    # model bytes or private dataset storage. JSON/logs remain directly browsable.
    checkpoints = sorted(stage.rglob('*.pth'))
    if checkpoints:
        with tarfile.open(stage / 'checkpoints.tar', 'w') as bundle:
            for path in checkpoints:
                bundle.add(path, arcname=str(path.relative_to(stage)))
        for path in checkpoints:
            path.unlink()
    try:
        kagglehub.dataset_upload(handle, str(stage), version_notes=note)
    finally:
        shutil.rmtree(stage)
    print(f'SECAP_ARCHIVED {handle}: {note}', flush=True)


def restore(handle):
    if list(ART.glob('*/progress.json')):
        print('Using the existing local Kaggle session checkpoints.', flush=True)
        return
    # Resolve the named archive directly. Recursive discovery across /kaggle/input
    # would traverse every source image folder again and delay session startup.
    import kagglehub
    print('SECAP_RESTORE_ARCHIVE', handle, flush=True)
    try:
        source = Path(kagglehub.dataset_download(handle))
    except Exception as e:
        if '404' not in str(e) and 'not found' not in str(e).lower():
            raise
        print('No previous archive: starting a new experiment.', flush=True)
        return
    previous_settings = source / 'run-settings.json'
    if previous_settings.exists():
        old = json.loads(previous_settings.read_text())
        new = json.loads((ART / 'run-settings.json').read_text())
        for key in ['batch_size', 'stride', 'max_epoch', 'eval_period']:
            # Older archive versions predate an explicit max_epoch setting.
            # Treat an absent key as historical metadata, not a checkpoint
            # incompatibility for a new (here, CARGO) experiment.
            if key in old and old.get(key) != new.get(key):
                raise RuntimeError(f'Resume setting changed: {key}; choose a new archive dataset for a new experiment.')
    # Kaggle expands uploaded tar archives into a directory of the same stem.
    if (source / 'checkpoints').is_dir():
        shutil.copytree(source / 'checkpoints', ART, dirs_exist_ok=True)
    if (source / 'checkpoints.tar').exists():
        with tarfile.open(source / 'checkpoints.tar') as bundle:
            bundle.extractall(ART, filter='data')
    for name in ['LAGPeR', 'CARGO', 'G2AVReID']:
        candidates = list(source.glob(f'**/{name}/progress.json'))
        if not candidates:
            continue
        src = candidates[0].parent
        dst = ART / name
        shutil.copytree(src, dst, dirs_exist_ok=True)
        info = json.loads((dst / 'progress.json').read_text())
        epoch_file = dst / f'model_epoch_{info["epoch"]:04d}.pth'
        last = dst / 'model_last.pth'
        if epoch_file.exists() and last.exists() and not os.path.samefile(epoch_file, last):
            last.unlink()
            os.link(epoch_file, last)
        best_epoch = info.get('best_epoch')
        if best_epoch is not None:
            best_epoch_file = dst / f'model_epoch_{best_epoch:04d}.pth'
            if best_epoch_file.exists():
                best = dst / 'model_best.pth'
                if not best.exists() or not os.path.samefile(best_epoch_file, best):
                    best.unlink(missing_ok=True)
                    os.link(best_epoch_file, best)
        if info.get('complete') and last.exists():
            final = dst / 'model_final.pth'
            final.unlink(missing_ok=True)
            os.link(last, final)
        (dst / 'last_checkpoint').write_text('model_last.pth')
        print('Restored', name, (dst / 'progress.json').read_text(), flush=True)


def make_plots():
    # Runs with training Python (where matplotlib is pinned).
    code = '''
import os
os.environ['MPLBACKEND'] = 'Agg'
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root = Path('/kaggle/working/secap-artifacts')
for p in root.glob('*/epochs.jsonl'):
    rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for k in sorted({k for row in rows for k in row['metrics']}):
        if 'loss' in k or k.endswith('/mAP') or k.endswith('/Rank-1') or k in ('mAP','Rank-1'):
            values=[(r['epoch'],r['metrics'][k]) for r in rows if k in r['metrics']]
            ax=axes[0] if 'loss' in k else axes[1]
            ax.plot(*zip(*values), label=k)
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.legend(fontsize=6); ax.grid(alpha=.3)
    fig.suptitle(p.parent.name); fig.tight_layout(); fig.savefig(p.parent/'training.png',dpi=160); plt.close(fig)
    progress=json.loads((p.parent/'progress.json').read_text())
    evaluations=[r for r in rows if r.get('evaluation')]
    summary={'dataset':p.parent.name,'progress':progress,'last_evaluation':evaluations[-1] if evaluations else None}
    (p.parent/'summary.json').write_text(json.dumps(summary,indent=2))
'''
    try:
        run([ENV / 'bin/python', '-c', code])
    except subprocess.CalledProcessError as error:
        (ART / 'plot-error.txt').write_text(str(error))
        print('SECAP_PLOT_ERROR: checkpoint archive will still be saved', flush=True)


def publish_heartbeat(stop, settings):
    import kagglehub
    folder = Path('/tmp/secap-heartbeat')
    folder.mkdir(exist_ok=True)
    while not stop.is_set():
        try:
            for name in settings['datasets']:
                src = ART / name
                dst = folder / name
                dst.mkdir(exist_ok=True)
                for filename in ['progress.json', 'best-evaluation.json', 'launch.json', 'epochs.jsonl']:
                    if (src / filename).exists():
                        shutil.copy2(src / filename, dst / filename)
                if (src / 'console.log').exists():
                    # Tail only, so monitor storage never grows with training length.
                    with (src / 'console.log').open('rb') as f:
                        f.seek(0, 2)
                        f.seek(max(0, f.tell() - 65536))
                        (dst / 'console-tail.log').write_bytes(f.read())
            write_json(folder / 'heartbeat.json', dict(time=time.time(), stage='training',
                       elapsed_hours=(time.time()-STARTED)/3600))
            kagglehub.dataset_upload(settings['monitor_dataset'], str(folder), version_notes='15-minute training heartbeat')
            print('SECAP_HEARTBEAT_PUBLISHED', flush=True)
        except Exception as e:
            print('SECAP_HEARTBEAT_ERROR', str(e)[:400], flush=True)
        stop.wait(900)


def quota_budget(settings):
    from kagglehub.clients import build_kaggle_client
    with build_kaggle_client() as client:
        quota = client.kernels.kernels_api_client.get_accelerator_quota_statistics()
    remaining = (quota.gpu_quota.total_time_allowed - quota.gpu_quota.time_used).total_seconds()
    budget = min(settings['session_hours'] * 3600, remaining - 3600)
    write_json(ART / 'gpu-quota.json', dict(allowed_hours=quota.gpu_quota.total_time_allowed.total_seconds()/3600,
               used_hours=quota.gpu_quota.time_used.total_seconds()/3600,
               training_budget_hours=budget/3600, refresh_time=str(quota.quota_refresh_time)))
    print('GPU quota and budget:', (ART / 'gpu-quota.json').read_text(), flush=True)
    if budget < 1800:
        raise RuntimeError('Insufficient GPU quota with archival reserve; retry after quota refresh.')
    return STARTED + budget


def main(settings):
    ART.mkdir(parents=True, exist_ok=True)
    handle = settings['artifact_dataset']
    write_json(ART / 'run-settings.json', settings)
    write_json(ART / 'source-manifest.json', dict(sha256=hashlib.file_digest(open('/tmp/secap-source.tar.gz', 'rb'), 'sha256').hexdigest(), author_commit=settings.get('source_commit')))
    # Preserve source and all adapters in the archive, not the original dataset.
    shutil.copy2('/tmp/secap-source.tar.gz', ART / 'source.tar.gz')
    prepare_data()
    py = install_environment()
    quarantine_cargo(py)
    run([py, 'kaggle/check_checkpoint.py'], cwd=REPO)
    run([py, 'kaggle/check_g2a.py'], cwd=REPO)
    weights = pretrained()
    restore(handle)
    # Verify automatic notebook authentication can archive BEFORE long training.
    # Kagglehub creates PRIVATE datasets by default. No token is embedded in code.
    snapshot_upload(handle, 'Preflight: dataset validation, environment and exact source')
    deadline = quota_budget(settings)
    os.environ['SECAP_DEADLINE'] = str(deadline)
    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(target=publish_heartbeat, args=(heartbeat_stop, settings), daemon=True)
    heartbeat.start()
    archive_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = None
    last_archive = time.time()
    try:
        for name in settings['datasets']:
            progress = ART / name / 'progress.json'
            if progress.exists() and json.loads(progress.read_text()).get('complete'):
                print(name, 'already complete', flush=True)
                continue
            if time.time() >= deadline:
                break
            output = ART / name
            output.mkdir(exist_ok=True)
            cmd = [str(py), 'kaggle/train_wrapper.py', '--config-file', f'configs/{name if name != "G2AVReID" else "LAGPeR"}/secap.yml',
                   '--num-gpus', '2', '--resume',
                   'MODEL.BACKBONE.PRETRAIN_PATH', str(weights), 'OUTPUT_DIR', str(output),
                   'SOLVER.IMS_PER_BATCH', str(settings['batch_size']),
                   'SOLVER.MAX_EPOCH', str(settings['max_epoch']),
                   'MODEL.BACKBONE.STRIDE_SIZE', str(tuple(settings['stride'])),
                   'DATALOADER.NUM_WORKERS', '2', 'TEST.IMS_PER_BATCH', '64',
                   'TEST.EVAL_PERIOD', str(settings['eval_period'])]
            if name == 'G2AVReID':
                cmd += ['DATASETS.NAMES', '("G2AVReID",)', 'DATASETS.TESTS', '("G2AVReID",)']
            elif name == 'CARGO':
                # Preserve the authors' overall/AG/AA/GG protocol and add the
                # reverse ground-to-aerial direction as a supplementary result.
                cmd += ['DATASETS.TESTS', '("CARGO", "CARGO_AG", "CARGO_GA", "CARGO_AA", "CARGO_GG")']
            write_json(output / 'launch.json', dict(command=cmd, seed=0,
                       distributed_note='Author DDP: triplet mining and BatchNorm use per-GPU batches; two GPUs differ from single-GPU paper training.'))
            with (output / 'console.log').open('a', buffering=1) as log:
                proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1)
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    if future is not None and future.done():
                        try:
                            future.result()
                        except Exception as e:
                            print('SECAP_ARCHIVE_RETRY_NEEDED', str(e)[:400], flush=True)
                        future = None
                    if time.time() - last_archive >= settings['archive_minutes'] * 60 and future is None:
                        future = archive_pool.submit(snapshot_upload, handle, f'Periodic snapshot: {name}')
                        last_archive = time.time()
                rc = proc.wait()
            if rc:
                raise RuntimeError(f'{name} training process exited {rc}; see {output}/console.log')
            if future is not None:
                future.result()
                future = None
            make_plots()
            snapshot_upload(handle, f'{name}: epoch checkpoint and metrics')
            if not progress.exists() or not json.loads(progress.read_text()).get('complete'):
                break
    except BaseException:
        (ART / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=60)
        archive_pool.shutdown(wait=True)
        try:
            make_plots()
        finally:
            snapshot_upload(handle, 'Session end: durable resumable artifacts')
    write_json(ART / 'session-end.json', dict(time=time.time(), elapsed_hours=(time.time()-STARTED)/3600))
    print('SECAP_SESSION_FINISHED: artifact dataset', handle, flush=True)
