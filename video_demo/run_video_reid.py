"""Detector + tracker + SeCap feature consistency demo for a pedestrian video.

Track IDs are temporal tracker identities.  They are not claims about a
person's real-world identity unless a labelled enrollment gallery is supplied.
"""
import argparse, csv, json, os, time
from pathlib import Path
os.environ['MPLBACKEND'] = 'Agg'
import cv2, numpy as np, torch
import torch.nn.functional as F


def load_reid(config, checkpoint):
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from fastreid.config import get_cfg
    from fastreid.modeling import build_model
    from fastreid.utils.checkpoint import Checkpointer
    cfg = get_cfg(); cfg.merge_from_file(str(config)); cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False; cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'; cfg.freeze()
    model = build_model(cfg).eval(); Checkpointer(model).load(str(checkpoint)); return model, cfg


def feature(model, image, size):
    rgb = cv2.cvtColor(cv2.resize(image, tuple(size[::-1]), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    ten = torch.from_numpy(rgb.astype('float32').transpose(2,0,1).copy()).unsqueeze(0).to(next(model.parameters()).device)
    with torch.inference_mode(): return F.normalize(model({'images': ten}), dim=1).cpu()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--csv',required=True); p.add_argument('--conf',type=float,default=.25); args=p.parse_args()
    from ultralytics import YOLO
    reid,cfg=load_reid(args.config,args.checkpoint); detector=YOLO('yolo11n.pt')
    cap=cv2.VideoCapture(args.input); fps=cap.get(cv2.CAP_PROP_FPS) or 30; w,h=int(cap.get(3)),int(cap.get(4))
    out=cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h)); refs={}; seen=set(); frame=0; ema=0.; rows=[]
    while True:
        ok,img=cap.read()
        if not ok: break
        started=time.perf_counter(); result=detector.track(img,persist=True,classes=[0],conf=args.conf,verbose=False)[0]
        detections=0
        if result.boxes is not None and result.boxes.id is not None:
            xyxy=result.boxes.xyxy.cpu().numpy().astype(int); ids=result.boxes.id.cpu().numpy().astype(int); conf=result.boxes.conf.cpu().numpy()
            for box,tid,score in zip(xyxy,ids,conf):
                x1,y1,x2,y2=np.clip(box,[0,0,0,0],[w-1,h-1,w-1,h-1]); crop=img[y1:y2,x1:x2]
                if crop.size==0: continue
                emb=feature(reid,crop,cfg.INPUT.SIZE_TEST); refs.setdefault(int(tid),emb)
                sim=float((emb@refs[int(tid)].T).item()); seen.add(int(tid)); detections+=1
                color=(0,220,0) if sim>=.55 else (0,165,255)
                cv2.rectangle(img,(x1,y1),(x2,y2),color,2); cv2.putText(img,f'Track {tid} | reID {sim:.2f}',(x1,max(20,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.52,color,2)
                rows.append({'frame':frame,'track_id':int(tid),'detector_confidence':float(score),'reference_similarity':sim,'x1':int(x1),'y1':int(y1),'x2':int(x2),'y2':int(y2)})
        elapsed=time.perf_counter()-started; ema=elapsed if not ema else .9*ema+.1*elapsed
        cv2.rectangle(img,(8,8),(390,105),(0,0,0),-1)
        cv2.putText(img,'SeCap video ReID demo',(18,32),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        cv2.putText(img,f'frame {frame} | live FPS {1/max(ema,1e-6):.1f}',(18,57),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        cv2.putText(img,f'persons {detections} | tracks seen {len(seen)}',(18,80),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        cv2.putText(img,'reID = similarity to first crop in this track',(18,100),cv2.FONT_HERSHEY_SIMPLEX,.42,(200,200,200),1)
        out.write(img); frame+=1
    cap.release(); out.release()
    with open(args.csv,'w',newline='') as f: csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ['frame']).writeheader(); [csv.DictWriter(f,fieldnames=rows[0].keys()).writerow(r) for r in rows] if rows else None
    Path(args.csv).with_suffix('.json').write_text(json.dumps({'frames':frame,'tracks_seen':len(seen),'observations':len(rows),'identity_claim':'tracker IDs with same-track SeCap reference similarity; no external identity ground truth'},indent=2))
if __name__=='__main__': main()
