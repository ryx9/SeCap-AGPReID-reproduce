# Video demonstrations

## MARS labelled cross-camera ReID

`mars-v1/` is the primary ReID demonstration. MARS provides known person IDs
and camera labels, so its two videos show a genuine cross-camera query/gallery
retrieval decision for each checkpoint. The result is reported in
`mars_reid_report.json` on a controlled 60-identity subset (not the official
full MARS leaderboard protocol).

| checkpoint | Rank-1 | mAP | displayed hard query |
|---|---:|---:|---|
| CARGO | 51.7% | 66.7% | PID 0106, rank 33/60 |
| LAGPeR | 73.3% | 81.2% | PID 0078, rank 29/60 |

## VisDrone human ground-truth demo

`visdrone-human-v6/` validates person detection/tracking against annotated
video. It is **not** a cross-camera identity benchmark. Its ground-truth
report records precision 79.7%, recall 65.8%, F1 72.1%, and mean dominant
tracker-ID consistency 95.9%.

## Qualitative single-video demo

`qualitative-v3/` is retained as a compact qualitative detector/tracker and
same-track similarity example. It has no external identity ground truth.
