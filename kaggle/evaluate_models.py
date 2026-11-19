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
    import kagglehub
    root = Path(kagglehub.dataset_download(handle))
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
        wrong = int(keep[0])
        true = int(keep[correct[0]])
        rows.append(dict(query_index=i, pid=int(qp[i]), rank=rank, ap=float(aps[len(rows)]),
                         query=paths[i], wrong=paths[num_query + wrong], true=paths[num_query + true],
                         wrong_pid=int(gp[wrong]), true_pid=int(gp[true])))
    return scores, rows


def select_distinct(rows, mode, count=30):
    ordered = sorted(rows, key=lambda x: (-x['rank'], x['ap'])) if mode == 'worst' else sorted(rows, key=lambda x: (x['pid'], x['query']))
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
