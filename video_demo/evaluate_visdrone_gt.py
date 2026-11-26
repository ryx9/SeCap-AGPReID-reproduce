"""Evaluate tracker boxes/IDs against VisDrone MOT ground truth."""
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0

def main():
 p=argparse.ArgumentParser();p.add_argument('--annotations',required=True);p.add_argument('--csv',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 gt=defaultdict(list)
 for line in Path(a.annotations).read_text().splitlines():
  f,tid,x,y,w,h,score,cls,*_=map(int,line.split(','))
  if cls in (1,2) and score: gt[f-1].append((tid,(x,y,x+w,y+h)))
 pred=defaultdict(list)
 for r in csv.DictReader(open(a.csv)):
  pred[int(r['frame'])].append((int(r['track_id']),(int(r['x1']),int(r['y1']),int(r['x2']),int(r['y2']))))
 tp=fp=fn=0; mapping=defaultdict(Counter)
 for f in set(gt)|set(pred):
  used=set()
  for pt,pb in pred[f]:
   choices=[(iou(pb,gb),i,gid) for i,(gid,gb) in enumerate(gt[f]) if i not in used]
   best=max(choices,default=(0,-1,-1))
   if best[0]>=.5: tp+=1;used.add(best[1]);mapping[pt][best[2]]+=1
   else: fp+=1
  fn+=len(gt[f])-len(used)
 precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0
 consistency={str(k):v.most_common(1)[0][1]/sum(v.values()) for k,v in mapping.items()}
 Path(a.out).write_text(json.dumps({'iou_threshold':.5,'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0,'track_id_to_gt_id_consistency':consistency,'note':'GT checks detection/tracker ID agreement; SeCap similarity remains same-track appearance only.'},indent=2))
if __name__=='__main__': main()
