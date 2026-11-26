import base64,hashlib,io,json,subprocess,tarfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]; O=R/'kaggle/visdrone_gt_notebook'; O.mkdir(exist_ok=True)
raw=subprocess.check_output(['git','archive','HEAD','fastreid','configs','LICENSE'],cwd=R); b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as d:
 with tarfile.open(fileobj=io.BytesIO(raw)) as s:
  for m in s:
   if not m.name.endswith(('.so','.pyc','.c')):d.addfile(m,s.extractfile(m) if m.isfile() else None)
 for x in ['kaggle/runtime.py','kaggle/requirements-training.txt','video_demo/run_video_reid.py','video_demo/evaluate_visdrone_gt.py']:d.add(R/x,arcname=x)
p=b.getvalue(); e=base64.b64encode(p).decode(); h=hashlib.sha256(p).hexdigest()
code=f'''import base64,hashlib,io,tarfile,subprocess
from pathlib import Path
p=base64.b64decode({e!r});assert hashlib.sha256(p).hexdigest()=={h!r};repo=Path('/tmp/secap-gt');repo.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(p)) as z:z.extractall(repo,filter='data')
ns={{'__name__':'runtime'}};exec(compile((repo/'kaggle/runtime.py').read_text(),'runtime.py','exec'),ns);ns['REPO']=repo;ns['ART']=Path('/kaggle/working/runtime');ns['ART'].mkdir(exist_ok=True);py=ns['install_environment']();ns['run'](['uv','pip','install','--python',py,'ultralytics==8.3.0','lapx==0.5.11'])
seq=next(Path('/kaggle/input').rglob('uav0000305_00000_v'));ann=next(Path('/kaggle/input').rglob('uav0000305_00000_v.txt'));video=Path('/kaggle/working/visdrone305.mp4');ns['run'](['ffmpeg','-y','-framerate','20','-i',seq/'%07d.jpg','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2','-c:v','libx264','-pix_fmt','yuv420p',video])
ck=next(Path('/kaggle/input').rglob('checkpoints/CARGO/model_best.pth'));root=ck.parents[2];out=Path('/kaggle/working/secap_visdrone_gt');out.mkdir(exist_ok=True);ns['run']([py,'video_demo/run_video_reid.py','--input',video,'--output',out/'annotated_visdrone.mp4','--config',root/'CARGO/config.yaml','--checkpoint',ck,'--csv',out/'track_reid.csv'],cwd=repo);ns['run']([py,'video_demo/evaluate_visdrone_gt.py','--annotations',ann,'--csv',out/'track_reid.csv','--out',out/'ground_truth_metrics.json'],cwd=repo)
'''
def c(t,s):return {'cell_type':t,'metadata':{},'source':s.splitlines(keepends=True),**({'execution_count':None,'outputs':[]} if t=='code' else {})}
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[c('markdown','# SeCap VisDrone ground-truth video test'),c('code',code)]};(O/'secap_visdrone_gt.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
meta={'id':'rizwanabdullah05/secap-visdrone-ground-truth-video','title':'SeCap VisDrone Ground Truth Video','code_file':'secap_visdrone_gt.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':['rizwanabdullah05/secap-reproduction-artifacts','nandhanivasmalasani/visdrone2019-mot-val'],'competition_sources':[],'kernel_sources':[],'model_sources':[]};(O/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
