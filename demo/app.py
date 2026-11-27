#!/usr/bin/env python3
"""Local SeCap app: gallery retrieval, annotated video, and live cameras."""
import argparse, io, json, re, secrets, shutil, sys, threading, time, traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT=Path(__file__).resolve().parents[1]; DEMO=Path(__file__).resolve().parent
STATIC=DEMO/'web'; OUT=DEMO/'outputs'; EXT={'.jpg','.jpeg','.png','.webp','.bmp'}

class ReID:
 def __init__(self,cfg,ck): self.cfg_path,self.ck,self.model,self.lock=cfg,ck,None,threading.Lock()
 def load(self):
  if self.model:return
  if not self.cfg_path.is_file() or not self.ck.is_file(): raise RuntimeError('Missing model artifacts. Run `uv run python demo/setup_artifacts.py`.')
  if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
  import numpy as np,torch
  from PIL import Image
  from fastreid.config import get_cfg
  from fastreid.modeling import build_model
  from fastreid.utils.checkpoint import Checkpointer
  cfg=get_cfg();cfg.merge_from_file(str(self.cfg_path));cfg.defrost();cfg.MODEL.BACKBONE.PRETRAIN=False;cfg.MODEL.DEVICE='cuda' if torch.cuda.is_available() else 'cpu';cfg.freeze()
  self.np,self.torch,self.Image,self.cfg=np,torch,Image,cfg;self.model=build_model(cfg).eval();Checkpointer(self.model).load(str(self.ck))
 def embed(self,images):
  with self.lock:
   self.load(); xs=[]
   for im in images:
    im=im[:,:,::-1];im=self.np.asarray(self.Image.fromarray(im).resize((self.cfg.INPUT.SIZE_TEST[1],self.cfg.INPUT.SIZE_TEST[0]),self.Image.Resampling.BICUBIC),dtype='float32');xs.append(self.torch.from_numpy(im.transpose(2,0,1).copy()))
   with self.torch.inference_mode(): f=self.model({'images':self.torch.stack(xs).to(next(self.model.parameters()).device)})
   return self.torch.nn.functional.normalize(f,dim=1).cpu()
 def paths(self,paths):
  import cv2
  out=[]
  for n in range(0,len(paths),24):
   good=[(p,cv2.imread(str(p))) for p in paths[n:n+24]];good=[x for x in good if x[1] is not None]
   if good: out+=list(zip([x[0] for x in good],self.embed([x[1] for x in good])))
  return out

class Gallery:
 def __init__(self,reid):self.reid,self.cache,self.lock=reid,{},threading.Lock()
 def get(self,raw,limit=2000):
  root=Path(raw).expanduser().resolve()
  if not root.exists():raise ValueError(f'Gallery does not exist: {root}')
  files=[root] if root.is_file() else sorted(x for x in root.rglob('*') if x.suffix.lower() in EXT)[:limit]
  if not files:raise ValueError('Gallery contains no supported images.')
  key=(str(root),len(files),max(x.stat().st_mtime_ns for x in files))
  with self.lock:
   if key not in self.cache:
    pairs=self.reid.paths(files);self.cache={key:([x[0] for x in pairs],self.reid.torch.stack([x[1] for x in pairs]))}
   return self.cache[key]

class Detector:
 def __init__(self,w):self.w,self.model,self.lock=w,None,threading.Lock()
 def run(self,frame):
  with self.lock:
   if self.model is None:
    from ultralytics import YOLO
    self.model=YOLO(self.w)
   r=self.model(frame,classes=[0],conf=.25,verbose=False)[0]
  return [] if r.boxes is None else [(tuple(map(int,b)),float(c)) for b,c in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy())]

JOBS={};JLOCK=threading.Lock()
def new_job(**d):
 ident=secrets.token_urlsafe(8);d.update(status='queued',stop=threading.Event())
 with JLOCK:JOBS[ident]=d
 return ident
def job(i):
 with JLOCK:return JOBS.get(i)
def images(raw,limit=100):
 p=Path(raw).expanduser().resolve()
 if not p.exists():raise ValueError(f'Path does not exist: {p}')
 return [p] if p.is_file() else sorted(x for x in p.rglob('*') if x.suffix.lower() in EXT)[:limit]
def fields(body,ctype):
 m=re.search(r'boundary=(?:"([^"]+)"|([^;\s]+))',ctype or '')
 if not m:raise ValueError('Expected a form submission.')
 ans={};b=(m.group(1) or m.group(2)).encode()
 for part in body.split(b'--'+b):
  h,sep,v=part.lstrip(b'\r\n').partition(b'\r\n\r\n');n=re.search(rb'name="([^"]+)"',h)
  if n and sep:ans[n.group(1).decode()]=v.rstrip(b'\r\n')
 return ans

def recognize(frame,gallery,model):
 import cv2
 reid=MODELS[model];files,feat=GALLERIES[model].get(gallery);boxes=DETECTOR.run(frame);crops=[];keep=[];h,w=frame.shape[:2]
 for (x1,y1,x2,y2),conf in boxes:
  x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2);crop=frame[y1:y2,x1:x2]
  if crop.size:crops.append(crop);keep.append(((x1,y1,x2,y2),conf))
 if not crops:return []
 out=[]
 for (box,conf),emb in zip(keep,reid.embed(crops)):
  scores=(emb.unsqueeze(0)@feat.T).squeeze();i=int(reid.torch.argmax(scores));out.append((box,files[i],float(scores[i])))
 return out
def footage_reid(frame,model,state):
 """Assign persistent local identities from this footage; no gallery required."""
 reid=MODELS[model];boxes=DETECTOR.run(frame);crops=[];keep=[];h,w=frame.shape[:2]
 for (x1,y1,x2,y2),conf in boxes:
  x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2);crop=frame[y1:y2,x1:x2]
  if crop.size:crops.append(crop);keep.append((x1,y1,x2,y2))
 if not crops:return []
 output=[]
 for box,emb in zip(keep,reid.embed(crops)):
  if state['refs']:
   scores=reid.torch.stack(state['refs'])@emb;best=int(reid.torch.argmax(scores));score=float(scores[best])
  else: best,score=-1,-1.0
  if score < .55:
   state['refs'].append(emb);best=len(state['refs'])-1;score=1.0
  output.append((box,f'Person {best+1}',score))
 return output
def draw(frame,hits,start,n):
 import cv2
 for (x1,y1,x2,y2),p,s in hits:
  col=(0,210,0) if s>=.55 else (0,165,255);cv2.rectangle(frame,(x1,y1),(x2,y2),col,2);cv2.putText(frame,f'{p} | ReID {s:.2f}',(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,col,2)
 cv2.rectangle(frame,(8,8),(520,67),(0,0,0),-1);cv2.putText(frame,f'SeCap ReID | people {len(hits)} | FPS {n/max(time.time()-start,.001):.1f}',(18,35),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2);cv2.putText(frame,'label = best gallery identity / cosine similarity',(18,58),cv2.FONT_HERSHEY_SIMPLEX,.43,(220,220,220),1)
def process(i,live=False):
 import cv2
 j=job(i);j['status']='running';src=int(j['source']) if live and j['source'].isdigit() else j['source'];cap=cv2.VideoCapture(src)
 if not cap.isOpened():j.update(status='failed',error=f'Cannot open source: {src}');j['stop'].set();return
 fps=cap.get(cv2.CAP_PROP_FPS) or 25;w,h=int(cap.get(3)),int(cap.get(4));total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));writer=None
 if not live:
  target=OUT/'videos'/f'{i}.mp4';target.parent.mkdir(parents=True,exist_ok=True);writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h));j['output']=str(target)
 start=time.time();n=0
 try:
  while not j['stop'].is_set():
   ok,frame=cap.read()
   if not ok:break
   # Publish a frame before the first detector/model load so the browser never
   # appears stuck while CUDA weights initialize.
   if live and n == 0:
    preview=frame.copy();cv2.putText(preview,'Initializing detector and SeCap ReID…',(18,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
    ok,buf=cv2.imencode('.jpg',preview,[cv2.IMWRITE_JPEG_QUALITY,85]);j['jpeg']=buf.tobytes() if ok else None;j['progress']='initializing model'
   hit=footage_reid(frame,j['model'],j['tracks']);draw(frame,hit,start,n+1);n+=1;j['progress']=f'{n}/{total or "?"} frames'
   if live:
    ok,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85]);j['jpeg']=buf.tobytes() if ok else None
   else:writer.write(frame)
  j['status']='stopped' if j['stop'].is_set() else 'complete'
 except Exception as e:
  traceback.print_exc();j.update(status='failed',error=str(e));j['stop'].set()
 finally:
  cap.release()
  if writer:writer.release()

class App(SimpleHTTPRequestHandler):
 def __init__(self,*a,**kw):super().__init__(*a,directory=str(STATIC),**kw)
 def reply(self,status,data):
  raw=json.dumps(data).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 def form(self):
  n=int(self.headers.get('Content-Length','0'))
  if not 0<n<=512*1024*1024:raise ValueError('Upload must be below 512 MB.')
  return fields(self.rfile.read(n),self.headers.get('Content-Type'))
 def do_GET(self):
  u=urlparse(self.path);q=parse_qs(u.query)
  if u.path=='/api/health':return self.reply(200,{'models':{k:{'checkpoint':str(v.ck),'loaded':bool(v.model)} for k,v in MODELS.items()}})
  if u.path=='/api/cameras':
   import cv2
   found=[]
   for index in range(8):
    cap=cv2.VideoCapture(index)
    if cap.isOpened():
     ok,_=cap.read()
     if ok:found.append({'source':str(index),'name':f'Local webcam {index}'})
    cap.release()
   return self.reply(200,{'cameras':found,'note':'Attached webcams are detected automatically. Add RTSP URLs manually.'})
  if u.path=='/api/job':
   j=job(q.get('id',[''])[0]);return self.reply(200,{k:v for k,v in j.items() if k not in ('stop','jpeg','tracks')} if j else {'error':'Unknown job'})
  if u.path.startswith('/api/download/'):
   j=job(u.path.rsplit('/',1)[-1]);p=Path(j.get('output','')) if j else None
   if not p or not p.is_file():return self.reply(404,{'error':'Output is not ready.'})
   self.send_response(200);self.send_header('Content-Type','video/mp4');self.send_header('Content-Length',str(p.stat().st_size));self.end_headers();shutil.copyfileobj(p.open('rb'),self.wfile);return
  if u.path.startswith('/api/live/'):
   j=job(u.path.split('/')[3]);
   if not j:return self.reply(404,{'error':'Unknown camera.'})
   self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=f');self.end_headers()
   while not j['stop'].is_set() and j.get('status') in ('queued','running'):
    if j.get('jpeg'):
     self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\n\r\n'+j['jpeg']+b'\r\n');self.wfile.flush()
    time.sleep(.08)
   return
  if u.path=='/':self.path='/index.html'
  return super().do_GET()
 def do_POST(self):
  try:
   f=self.form()
   if self.path=='/api/search':
    name=f.get('model',b'cargo').decode().lower();reid=MODELS.get(name)
    if not reid:raise ValueError('Choose CARGO or LAGPeR.')
    qs=images(f.get('query_path',b'').decode());files,feat=GALLERIES[name].get(f.get('gallery_path',b'').decode(),int(f.get('max_gallery',b'2000')));r=[]
    for p,e in reid.paths(qs):
     s=(e@feat.T).squeeze();idx=reid.torch.argsort(s,descending=True)[:10].tolist();r.append({'query':str(p),'matches':[{'path':str(files[x]),'score':round(float(s[x]),4)} for x in idx]})
    return self.reply(200,{'gallery_images':len(files),'results':r})
   if self.path in ('/api/video','/api/live'):
    sources=[x.strip() for x in f.get('sources',f.get('source',b'')).decode().splitlines() if x.strip()];model=f.get('model',b'cargo').decode().lower()
    if model not in MODELS:raise ValueError('Choose CARGO or LAGPeR.')
    if not sources:raise ValueError('Provide a video path or one camera source per line.')
    ids=[]
    for src in sources:
     i=new_job(source=src,model=model,tracks={'refs':[]},kind='live' if self.path.endswith('live') else 'video');ids.append(i);threading.Thread(target=process,args=(i,self.path.endswith('live')),daemon=True).start()
    return self.reply(202,{'ids':ids})
   if self.path=='/api/stop':
    j=job(f.get('id',b'').decode());j['stop'].set() if j else None;return self.reply(200,{'stopped':bool(j)})
   self.reply(404,{'error':'Unknown endpoint.'})
  except Exception as e:traceback.print_exc();self.reply(400,{'error':str(e)})
 def log_message(self,fmt,*a):print('[web] '+fmt%a)

def main():
 p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8000);p.add_argument('--cargo-config',type=Path,default=DEMO/'artifacts/cargo/config.yaml');p.add_argument('--cargo-checkpoint',type=Path,default=DEMO/'artifacts/cargo/model_best.pth');p.add_argument('--lagper-config',type=Path,default=DEMO/'artifacts/lagper/config.yaml');p.add_argument('--lagper-checkpoint',type=Path,default=DEMO/'artifacts/lagper/model_best.pth');p.add_argument('--detector',default='yolo11n.pt');a=p.parse_args()
 global MODELS,DETECTOR,GALLERIES;MODELS={'cargo':ReID(a.cargo_config,a.cargo_checkpoint),'lagper':ReID(a.lagper_config,a.lagper_checkpoint)};DETECTOR=Detector(a.detector);GALLERIES={k:Gallery(v) for k,v in MODELS.items()};OUT.mkdir(exist_ok=True);print(f'http://{a.host}:{a.port}');ThreadingHTTPServer((a.host,a.port),App).serve_forever()
if __name__=='__main__':main()
