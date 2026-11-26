"""Build private Kaggle video detector/tracker/SeCap demo."""
import base64, hashlib, io, json, subprocess, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'kaggle/video_demo_notebook'; OUT.mkdir(exist_ok=True)
raw=subprocess.check_output(['git','archive','HEAD','fastreid','configs','LICENSE'],cwd=ROOT); blob=io.BytesIO()
with tarfile.open(fileobj=blob,mode='w:gz') as dst:
    with tarfile.open(fileobj=io.BytesIO(raw)) as src:
        for m in src:
            if not m.name.endswith(('.so','.pyc','.c')): dst.addfile(m,src.extractfile(m) if m.isfile() else None)
    dst.add(ROOT/'kaggle/runtime.py',arcname='kaggle/runtime.py'); dst.add(ROOT/'kaggle/requirements-training.txt',arcname='kaggle/requirements-training.txt'); dst.add(ROOT/'video_demo/run_video_reid.py',arcname='video_demo/run_video_reid.py')
payload=blob.getvalue(); encoded=base64.b64encode(payload).decode(); digest=hashlib.sha256(payload).hexdigest()
code=f'''import base64,hashlib,io,tarfile,os,urllib.request
from pathlib import Path
p=base64.b64decode({encoded!r}); assert hashlib.sha256(p).hexdigest()=={digest!r}
repo=Path('/tmp/secap-video'); repo.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(p)) as z: z.extractall(repo,filter='data')
ns={{'__name__':'runtime'}}; exec(compile((repo/'kaggle/runtime.py').read_text(),'runtime.py','exec'),ns)
ns['REPO']=repo; ns['ART']=Path('/kaggle/working/secap-video-runtime'); ns['ART'].mkdir(exist_ok=True)
py=ns['install_environment'](); ns['run'](['uv','pip','install','--python',py,'ultralytics==8.3.0','lapx==0.5.11'])
roots=list(Path('/kaggle/input').rglob('checkpoints/CARGO/model_best.pth')); assert roots, 'Attach secap-reproduction-artifacts'
archive=roots[0].parents[2]; video=Path('/kaggle/working/people-detection.mp4')
urllib.request.urlretrieve('https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4',video)
out=Path('/kaggle/working/secap_video_demo'); out.mkdir(exist_ok=True)
ns['run']([py,'video_demo/run_video_reid.py','--input',video,'--output',out/'annotated_secap_reid.mp4','--config',archive/'CARGO/config.yaml','--checkpoint',archive/'checkpoints/CARGO/model_best.pth','--csv',out/'track_reid_stats.csv'],cwd=repo)
'''
def c(t,s): return {'cell_type':t,'metadata':{},'source':s.splitlines(keepends=True),**({'execution_count':None,'outputs':[]} if t=='code' else {})}
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[c('markdown','# SeCap video tracking and ReID demo\n\nOutput video includes boxes, temporal tracker IDs, SeCap same-track reference similarity, live FPS, person count, and tracks seen.'),c('code',code)]}
(OUT/'secap_video_reid_demo.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
meta={'id':'rizwanabdullah05/secap-video-reid-demo','title':'SeCap Video ReID Demo','code_file':'secap_video_reid_demo.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':['rizwanabdullah05/secap-reproduction-artifacts'],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(OUT/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
