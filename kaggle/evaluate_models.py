"""Evaluate archived CARGO/LAGPeR best checkpoints and render retrieval audits.

Runs inside the Kaggle evaluation notebook.  It keeps all feature matrices,
reports and contact sheets in /kaggle/working, never in the input datasets.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tarfile

# The notebook invokes this file as ``python kaggle/evaluate_models.py``.
# In that form Python adds ``.../kaggle`` (rather than the repository root)
# to sys.path, so make the bundled FastReID package explicitly importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from fastreid.utils.checkpoint import Checkpointer
from fastreid.config import get_cfg
from fastreid.data import build_reid_test_loader
from fastreid.engine import DefaultTrainer
from fastreid.evaluation.rank import evaluate_rank


PROTOCOLS = {
    'CARGO': ('CARGO', 'CARGO_AG', 'CARGO_GA', 'CARGO_AA', 'CARGO_GG'),
    'LAGPeR': ('LAGPeR_A2G', 'LAGPeR_G2A', 'LAGPeR_G2AG'),
}
PRIMARY = {'CARGO': 'CARGO_AG', 'LAGPeR': 'LAGPeR_A2G'}


def archive_root(handle):
    """Locate the private artifacts input already attached to this kernel.

    Kaggle mounts private datasets below ``/kaggle/input``.  Searching the
    shallow directory structure avoids a Python-version-dependent client
    package in the repository's Python 3.10 environment.
    """
    root = None
    skip = {'bounding_box_train', 'bounding_box_test_aerial',
            'bounding_box_test_ground', 'query_aerial', 'query_ground',
            'bounding_box_gallery', 'bbox_train', 'bbox_test', 'train',
            'query', 'gallery'}
    for current, dirs, _ in os.walk('/kaggle/input'):
        candidate = Path(current)
        if (candidate / 'checkpoints').is_dir() or (candidate / 'checkpoints.tar').is_file():
            root = candidate
            break
        dirs[:] = [d for d in dirs if d not in skip]
    if root is None:
        raise FileNotFoundError(
            f'Could not find archived checkpoints under /kaggle/input; expected attached dataset {handle!r}.')
    print('SECAP_EVAL_ARCHIVE_ROOT', root, flush=True)
    # Kaggle normally expands tar files, but make the notebook portable.
    if not (root / 'checkpoints').exists() and (root / 'checkpoints.tar').exists():
        target = Path('/tmp/secap-eval-archive')
        target.mkdir(exist_ok=True)
        with tarfile.open(root / 'checkpoints.tar') as bundle:
            bundle.extractall(target, filter='data')
        for name in ('CARGO', 'LAGPeR'):
            if (root / name).exists():
                import shutil
                shutil.copytree(root / name, target / name, dirs_exist_ok=True)
        return target
    return root


def cfg_for(root, source, target):
    cfg = get_cfg()
    cfg.merge_from_file(str(root / source / 'config.yaml'))
    cfg.defrost()
    cfg.DATASETS.TESTS = (target,)
    cfg.TEST.IMS_PER_BATCH = 64
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.DEVICE = 'cuda'
    # A complete trained checkpoint is loaded immediately afterwards.  Do not
    # require the training-time ImageNet bootstrap file on this fresh kernel.
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.freeze()
    return cfg


def infer(model, loader):
    feats, pids, cams, paths = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            paths.extend(batch['img_paths'])
            pids.append(batch['targets'].cpu().numpy())
            cams.append(batch['camids'].cpu().numpy())
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.cuda(non_blocking=True)
            feats.append(model(batch).cpu())
    return torch.cat(feats), np.concatenate(pids), np.concatenate(cams), paths


def metrics_and_rows(feats, pids, cams, paths, num_query):
    qf, gf = feats[:num_query], feats[num_query:]
    qp, gp = pids[:num_query], pids[num_query:]
    qc, gc = cams[:num_query], cams[num_query:]
    dist = (1 - F.normalize(qf, dim=1) @ F.normalize(gf, dim=1).t()).numpy()
    cmc, aps, inps = evaluate_rank(dist, qp, gp, qc, gc)
    scores = {'Rank-1': float(cmc[0] * 100), 'Rank-5': float(cmc[4] * 100),
              'Rank-10': float(cmc[9] * 100), 'mAP': float(np.mean(aps) * 100),
              'mINP': float(np.mean(inps) * 100)}
    scores['metric'] = (scores['Rank-1'] + scores['mAP']) / 2
    rows = []
    order = np.argsort(dist, axis=1)
    for i, ranked in enumerate(order):
        keep = ranked[~((gp[ranked] == qp[i]) & (gc[ranked] == qc[i]))]
        correct = np.flatnonzero(gp[keep] == qp[i])
        if not len(correct):
            continue
        rank = int(correct[0]) + 1
        # `keep[0]` can itself be the correct identity.  It is valid for
        # ranking but must never be displayed as the "top incorrect" image.
        wrong_candidates = keep[gp[keep] != qp[i]]
        wrong = int(wrong_candidates[0]) if len(wrong_candidates) else int(keep[0])
        true = int(keep[correct[0]])
        rows.append(dict(query_index=i, pid=int(qp[i]), rank=rank, ap=float(aps[len(rows)]),
                         query=paths[i], wrong=paths[num_query + wrong], true=paths[num_query + true],
                         wrong_pid=int(gp[wrong]), true_pid=int(gp[true])))
    return scores, rows


def select_distinct(rows, mode, count=30):
    if mode == 'random':
        # One reproducibly sampled query for each sampled identity.  Keeping
        # the seed fixed makes the visual audit comparable across reruns.
        by_pid = {}
        for row in rows:
            by_pid.setdefault(row['pid'], []).append(row)
        rng = np.random.default_rng(20251119)
        pids = sorted(by_pid)
        rng.shuffle(pids)
        return [by_pid[pid][int(rng.integers(len(by_pid[pid])))] for pid in pids[:count]]
    ordered = sorted(rows, key=lambda x: (-x['rank'], x['ap']))
    chosen, seen = [], set()
    for row in ordered:
        if row['pid'] not in seen:
            chosen.append(row); seen.add(row['pid'])
        if len(chosen) == count:
            break
    return chosen


def contact_sheet(rows, destination, title):
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, max(8, 2.5 * len(rows))))
    for row, axis in zip(rows, axes):
        for ax, key, label in zip(axis, ('query', 'wrong', 'true'), ('query', 'top incorrect', 'first correct')):
            with Image.open(row[key]) as image:
                ax.imshow(image.convert('RGB'))
            ax.set_title(f'{label}\npid={row["pid"] if key=="query" else row[key+"_pid"] if key != "query" else row["pid"]}\nrank={row["rank"]}', fontsize=7)
            ax.axis('off')
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def run_pair(root, source, target, out):
    checkpoint = root / 'checkpoints' / source / 'model_best.pth'
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    pair = f'{source}_model__{target}_data'
    pair_out = out / pair
    pair_out.mkdir(parents=True, exist_ok=True)
    report = {'source_model': source, 'target_data': target, 'checkpoint': str(checkpoint), 'protocols': {}}
    for protocol in PROTOCOLS[target]:
        cfg = cfg_for(root, source, protocol)
        model = DefaultTrainer.build_model(cfg).cuda()
        Checkpointer(model).load(str(checkpoint))
        loader, num_query = build_reid_test_loader(cfg, dataset_name=protocol)
        feats, pids, cams, paths = infer(model, loader)
        scores, rows = metrics_and_rows(feats, pids, cams, paths, num_query)
        report['protocols'][protocol] = scores
        (pair_out / f'{protocol}_identity_failures.json').write_text(json.dumps(sorted(rows, key=lambda x: -x['rank']), indent=2))
        if protocol == PRIMARY[target]:
            contact_sheet(select_distinct(rows, 'random'), pair_out / f'{protocol}_random_30.png', f'{pair}: random distinct IDs')
            contact_sheet(select_distinct(rows, 'worst'), pair_out / f'{protocol}_worst_30.png', f'{pair}: worst distinct IDs')
            failures = sorted(rows, key=lambda x: -x['rank'])[:30]
            report['hardest_id_examples'] = [{k: r[k] for k in ('pid','rank','ap','query','wrong','true','wrong_pid')} for r in failures]
        del model
        torch.cuda.empty_cache()
    (pair_out / 'metrics.json').write_text(json.dumps(report, indent=2))
    return pair, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', default='rizwanabdullah05/secap-reproduction-artifacts')
    parser.add_argument('--output', default='/kaggle/working/secap-evaluation')
    args = parser.parse_args()
    root, out = archive_root(args.archive), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for source in ('CARGO', 'LAGPeR'):
        for target in ('CARGO', 'LAGPeR'):
            name, report = run_pair(root, source, target, out)
            summary[name] = report
            print('SECAP_EVAL_PAIR_COMPLETE', name, flush=True)
    (out / 'comprehensive_metrics.json').write_text(json.dumps(summary, indent=2))
    print('SECAP_EVALUATION_COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
