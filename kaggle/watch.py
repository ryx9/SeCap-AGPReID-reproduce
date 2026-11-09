"""Small local control plane: polls only metadata/logs; never downloads source images.
Run: uv run python kaggle/watch.py --loop
"""
import argparse
import json
from pathlib import Path
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from session_tools import observe
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest, ApiListTreeDatasetFilesRequest

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'kaggle/monitor'
KERNEL = 'rizwanabdullah05/secap-reproduction-t4x2'
DATASET = 'rizwanabdullah05/person-reid-cargo-g2a-lagper'
ARCHIVE = 'rizwanabdullah05/secap-reproduction-artifacts'
EXPERIMENTS = ('CARGO',)


def emit(message, **fields):
    row = dict(time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message=message, **fields)
    STATE.mkdir(exist_ok=True)
    with (STATE / 'events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def ready(api):
    with api.build_kaggle_client() as c:
        r = ApiGetDatasetRequest()
        r.owner_slug, r.dataset_slug = DATASET.split('/')
        data = c.datasets.dataset_api_client.get_dataset(r)
        status = data.versions[-1].status
        tree = ApiListTreeDatasetFilesRequest()
        tree.owner_slug, tree.dataset_slug = DATASET.split('/')
        tree.page_size = 30
        files = c.datasets.dataset_api_client.list_tree_dataset_files(tree)
        names = [d.name for d in files.directories]
        emit('dataset readiness', status=str(status), version=data.current_version_number,
             bytes=data.total_bytes, top_level=names)
        return str(status).lower() == 'ready' and 'CARGO' in names and 'LAGPeR' in names


def progress(api):
    result = {}
    for name in EXPERIMENTS:
        out = STATE / 'archive-progress'
        out.mkdir(exist_ok=True)
        try:
            api.dataset_download_file(ARCHIVE, name + '/progress.json', path=str(out), force=True, quiet=True, unzip=True)
            # CLI preserves subdirectory names for modern dataset files.
            candidates = list(out.rglob('progress.json'))
            for p in candidates:
                info = json.loads(p.read_text())
                if p.parent.name == name:
                    result[name] = info
                    break
            # Some API releases flatten individual downloads.
            flat = out / 'progress.json'
            if name not in result and flat.exists():
                result[name] = json.loads(flat.read_text())
                flat.unlink()
        except Exception as e:
            emit('archive progress unavailable', dataset=name, detail=str(e)[:300])
    return result


def check(api):
    status = api.kernels_status(KERNEL)
    label = str(status.status).rsplit('.', 1)[-1].lower()
    logs = api.kernels_logs(KERNEL)
    try:
        records = json.loads(logs)
        if isinstance(records, list):
            logs = ''.join(record.get('data', '') for record in records)
    except (ValueError, TypeError):
        pass
    if label == 'running' or not logs:
        try:
            live = observe(api, KERNEL, seconds=8)
            logs = live['logs'] or logs
            emit('live run identified', version=live['version'], run_id=live['run_id'])
        except Exception as e:
            emit('live log observation unavailable', detail=str(e)[:180])
    (STATE / 'latest-kernel.log').write_text(logs)
    try:
        api.dataset_download_file('rizwanabdullah05/secap-training-heartbeat', 'heartbeat.json', path=str(STATE / 'heartbeat'), force=True, quiet=True, unzip=True)
        heartbeat = json.loads((STATE / 'heartbeat/heartbeat.json').read_text())
        emit('remote training heartbeat', age_seconds=round(time.time() - heartbeat['time']), heartbeat=heartbeat)
        for name in EXPERIMENTS:
            try:
                api.dataset_download_file('rizwanabdullah05/secap-training-heartbeat', name + '/console-tail.log', path=str(STATE / 'heartbeat' / name), force=True, quiet=True, unzip=True)
            except Exception:
                pass
    except Exception:
        emit('remote heartbeat not yet available')
    emit('training check', status=label, failure=status.failure_message,
         log_bytes=len(logs), last_lines=logs.splitlines()[-12:])
    (STATE / 'status.json').write_text(json.dumps(dict(status=label, time=time.time(), failure=status.failure_message), indent=2))
    return label


def main(loop, resume_sessions, launch_when_stopped=False, interval=2700, not_before=0, wall_minute=None):
    STATE.mkdir(exist_ok=True)
    (STATE / 'monitor.pid').write_text(str(__import__('os').getpid()))
    if not_before > time.time():
        time.sleep(not_before - time.time())
    api = KaggleApi()
    api.authenticate()
    while not ready(api):
        if not loop:
            return
        time.sleep(300)
    if launch_when_stopped:
        emit('waiting for initial version to stop before submitting the corrected notebook')
        while True:
            try:
                current = str(api.kernels_status(KERNEL).status).rsplit('.', 1)[-1].lower()
                if current in ('cancelled', 'canceled', 'cancel_acknowledged', 'cancel_requested', 'error', 'complete', 'completed'):
                    result = api.kernels_push(str(ROOT / 'kaggle/notebook'), acc='NvidiaTeslaT4')
                    if result.error:
                        emit('corrected notebook submission failed', error=result.error)
                        return
                    emit('corrected notebook submitted', version=result.version_number)
                    break
            except Exception as e:
                emit('waiting for stop: API error', detail=str(e)[:300])
            time.sleep(30)
    while True:
        try:
            status = check(api)
            if status in ('complete', 'completed'):
                p = progress(api)
                emit('session completed', progress=p)
                if all(name in p and p[name].get('complete') for name in EXPERIMENTS):
                    emit('all requested experiments complete; fetching final models and results')
                    dest = ROOT / 'results/kaggle'
                    dest.mkdir(parents=True, exist_ok=True)
                    api.kernels_output(KERNEL, path=str(dest), file_pattern=r'.*(model_(best|last)\.pth|epochs\.jsonl|metrics\.json|progress\.json|config\.yaml|training\.png|.*\.log|.*\.json)$')
                    emit('final results downloaded', path=str(dest))
                    return
                if resume_sessions and p and (any(v.get('budget_stop') for v in p.values()) or all(v.get('complete') for v in p.values())):
                    # The next committed session auto-restores the archive, skips
                    # completed experiments and continues from the last epoch.
                    result = api.kernels_push(str(ROOT / 'kaggle/notebook'), acc='NvidiaTeslaT4')
                    if result.error:
                        emit('session continuation blocked', error=result.error)
                        return
                    emit('started next resumable session', version=result.version_number)
                else:
                    emit('session needs inspection; no automatic resubmission')
                    return
            elif status in ('error', 'cancelled', 'canceled'):
                emit('training failed; inspect latest-kernel.log before restarting')
                return
        except Exception as e:
            emit('monitor API error', detail=str(e)[:500])
        if not loop:
            return
        if wall_minute is None:
            time.sleep(interval)
        else:
            from datetime import datetime, timedelta
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo('Asia/Karachi'))
            target = now.replace(minute=wall_minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(hours=1)
            time.sleep(max(0, target.timestamp() - time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--resume-sessions', action='store_true')
    parser.add_argument('--launch-when-stopped', action='store_true', help='Submit the ready local notebook once the current version terminates, then monitor normally.')
    parser.add_argument('--interval', type=int, default=2700, help='Seconds between ongoing training checks; default 45 minutes.')
    parser.add_argument('--not-before', type=float, default=0)
    parser.add_argument('--wall-minute', type=int, choices=range(60))
    args = parser.parse_args()
    main(args.loop, args.resume_sessions, args.launch_when_stopped, args.interval, args.not_before, args.wall_minute)
