"""Create a ground-truth MARS video-ReID retrieval demonstration.

MARS consists of labelled person tracklets.  Each rendered frame shows an
identity-cropped query frame and cross-camera gallery candidates; the coloured
border is the original person bounding-box crop, not a detector prediction.
"""
import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import cv2
import numpy as np
import torch
import torch.nn.functional as F

NAME = re.compile(r"(?P<pid>.+?)C(?P<cam>\d+)T(?P<track>\d+)F\d+", re.I)


def load_reid(config, checkpoint):
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from fastreid.config import get_cfg
    from fastreid.modeling import build_model
    from fastreid.utils.checkpoint import Checkpointer
    cfg = get_cfg(); cfg.merge_from_file(str(config)); cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    model = build_model(cfg).eval()
    Checkpointer(model).load(str(checkpoint))
    return model, cfg


def discover(root, limit=60):
    """Return deterministic (pid, query-tracklet, gallery-tracklet) triples."""
    candidates = [p for p in root.rglob("bbox_test") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError("Could not find MARS bbox_test directory")
    # Kaggle mirrors sometimes add bbox_test/bbox_test; choose the leaf with ID dirs.
    leaf = max(candidates, key=lambda p: sum(x.is_dir() for x in p.iterdir()))
    per_pid = defaultdict(lambda: defaultdict(list))
    for id_dir in sorted(p for p in leaf.iterdir() if p.is_dir()):
        for image in sorted(id_dir.iterdir()):
            if image.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            m = NAME.match(image.stem)
            if m:
                per_pid[m["pid"]][(m["cam"], m["track"])].append(image)
    pairs = []
    for pid, tracks in sorted(per_pid.items()):
        usable = [(key, paths) for key, paths in sorted(tracks.items()) if len(paths) >= 4]
        for (q_key, q_paths) in usable:
            other = next(((g_key, g_paths) for g_key, g_paths in usable if g_key[0] != q_key[0]), None)
            if other:
                pairs.append({"pid": pid, "query": q_paths, "gallery": other[1],
                              "query_camera": q_key[0], "gallery_camera": other[0][0]})
                break
        if len(pairs) >= limit:
            break
    if len(pairs) < 8:
        raise RuntimeError(f"Only found {len(pairs)} cross-camera MARS identities under {leaf}")
    return pairs, leaf


def embed_tracklet(model, cfg, paths, sample=8):
    indices = np.linspace(0, len(paths) - 1, min(sample, len(paths)), dtype=int)
    images = []
    for i in indices:
        bgr = cv2.imread(str(paths[i]))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(cv2.resize(bgr, tuple(cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        images.append(rgb.astype("float32").transpose(2, 0, 1))
    ten = torch.from_numpy(np.stack(images)).to(next(model.parameters()).device)
    with torch.inference_mode():
        features = F.normalize(model({"images": ten}), dim=1)
    return F.normalize(features.mean(0, keepdim=True), dim=1).cpu()


def panel(canvas, image, x, y, w, h, caption, colour):
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    canvas[y:y+h, x:x+w] = image
    cv2.rectangle(canvas, (x, y), (x+w, y+h), colour, 4)
    cv2.putText(canvas, caption, (x, y+h+24), cv2.FONT_HERSHEY_SIMPLEX, .52, colour, 2)


def make_video(model_name, model, cfg, pairs, output, fps=12):
    gallery = torch.cat([embed_tracklet(model, cfg, pair["gallery"]) for pair in pairs])
    query = torch.cat([embed_tracklet(model, cfg, pair["query"]) for pair in pairs])
    sim = query @ gallery.T
    ranks, aps = [], []
    for i in range(len(pairs)):
        order = torch.argsort(sim[i], descending=True).tolist()
        rank = order.index(i) + 1
        ranks.append(rank); aps.append(1.0 / rank)
    # A hard, labelled sample reveals failure behavior; otherwise use median difficulty.
    chosen = max(range(len(pairs)), key=lambda i: ranks[i])
    pair, order = pairs[chosen], torch.argsort(sim[chosen], descending=True).tolist()
    rank = ranks[chosen]; top = order[:4]
    h, w = 720, 1280
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    ema = 0.0
    for frame_i, path in enumerate(pair["query"]):
        started = time.perf_counter()
        canvas = np.zeros((h, w, 3), np.uint8)
        canvas[:] = (26, 24, 22)
        correct = rank == 1
        verdict, verdict_colour = ("CORRECT", (0, 220, 0)) if correct else ("WRONG", (20, 70, 255))
        cv2.putText(canvas, "SeCap on MARS: labelled cross-camera video ReID", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        cv2.putText(canvas, f"model {model_name} | GT identity {pair['pid']} | query C{pair['query_camera']} -> gallery C{pair['gallery_camera']}", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, .57, (220, 220, 220), 1)
        img = cv2.imread(str(path))
        panel(canvas, img, 50, 120, 280, 450, f"query: PID {pair['pid']}  (GT box crop)", (0, 215, 255))
        cv2.putText(canvas, f"{verdict}: true match rank {rank}/{len(pairs)}", (405, 120), cv2.FONT_HERSHEY_SIMPLEX, .72, verdict_colour, 2)
        for j, idx in enumerate(top):
            candidate = cv2.imread(str(pairs[idx]["gallery"][len(pairs[idx]["gallery"]) // 2]))
            colour = (0, 220, 0) if idx == chosen else (20, 70, 255)
            panel(canvas, candidate, 405 + (j % 2) * 360, 155 + (j // 2) * 260, 190, 205,
                  f"rank {j+1}: PID {pairs[idx]['pid']}  sim {sim[chosen, idx]:.3f}", colour)
        elapsed = time.perf_counter() - started
        ema = elapsed if not ema else .9 * ema + .1 * elapsed
        cv2.rectangle(canvas, (25, 620), (1255, 705), (0, 0, 0), -1)
        cv2.putText(canvas, f"live frame {frame_i+1}/{len(pair['query'])} | render FPS {1/max(ema, 1e-6):.1f} | test queries {len(pairs)}", (42, 650), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        cv2.putText(canvas, f"benchmark Rank-1 {100*np.mean(np.array(ranks)==1):.1f}% | mAP {100*np.mean(aps):.1f}% | showing hardest query", (42, 682), cv2.FONT_HERSHEY_SIMPLEX, .58, (210, 210, 210), 1)
        writer.write(canvas)
    writer.release()
    return {"model": model_name, "benchmark": "MARS labelled cross-camera tracklets", "num_queries": len(pairs),
            "rank1": 100 * float(np.mean(np.array(ranks) == 1)), "mAP": 100 * float(np.mean(aps)),
            "shown_pid": pair["pid"], "shown_rank": rank, "shown_correct": bool(rank == 1),
            "query_camera": pair["query_camera"], "gallery_camera": pair["gallery_camera"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cargo-config", required=True); ap.add_argument("--cargo-checkpoint", required=True)
    ap.add_argument("--lagper-config", required=True); ap.add_argument("--lagper-checkpoint", required=True)
    ap.add_argument("--identities", type=int, default=60)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pairs, leaf = discover(Path(args.data), args.identities)
    results = {"source": str(leaf), "identities": len(pairs), "models": []}
    for name, config, checkpoint in [("CARGO", args.cargo_config, args.cargo_checkpoint), ("LAGPeR", args.lagper_config, args.lagper_checkpoint)]:
        model, cfg = load_reid(config, checkpoint)
        results["models"].append(make_video(name, model, cfg, pairs, out / f"mars_{name.lower()}_annotated.mp4"))
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    (out / "mars_reid_report.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
