"""Build the private Kaggle notebook for checkpoint evaluation and retrieval audits."""
import base64, hashlib, io, json, subprocess, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'kaggle/evaluation_notebook'
OUT.mkdir(exist_ok=True)
raw = subprocess.check_output(['git', 'archive', 'HEAD', 'fastreid', 'configs', 'tools', 'LICENSE'], cwd=ROOT)
blob = io.BytesIO()
with tarfile.open(fileobj=blob, mode='w:gz') as dest:
    with tarfile.open(fileobj=io.BytesIO(raw)) as src:
        for member in src:
            if not member.name.endswith(('.so', '.pyc', '.c')):
                dest.addfile(member, src.extractfile(member) if member.isfile() else None)
    for name in ['runtime.py', 'requirements-training.txt', 'evaluate_models.py']:
        dest.add(ROOT / 'kaggle' / name, arcname='kaggle/' + name)
payload = blob.getvalue()
encoded, digest = base64.b64encode(payload).decode(), hashlib.sha256(payload).hexdigest()

def cell(kind, source):
    data = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code': data.update(execution_count=None, outputs=[])
    return data

cells = [
    cell('markdown', '# SeCap comprehensive checkpoint evaluation\n\nEvaluates archived CARGO and LAGPeR best checkpoints in-domain and cross-dataset. It saves metrics, random 30-ID retrieval sheets, worst 30-ID retrieval sheets, and identity-level failure files.'),
    cell('code', f'''import base64, hashlib, io, tarfile
from pathlib import Path
payload = base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
repo = Path('/tmp/secap-source'); repo.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(payload)) as bundle:
    bundle.extractall(repo, filter='data')
namespace = {{'__name__': 'secap_eval_runtime'}}
exec(compile((repo/'kaggle/runtime.py').read_text(), 'kaggle/runtime.py', 'exec'), namespace)
namespace['ART'] = Path('/kaggle/working/secap-eval-runtime')
namespace['ART'].mkdir(exist_ok=True)
namespace['prepare_data']()
py = namespace['install_environment']()
namespace['run']([py, 'kaggle/evaluate_models.py'], cwd=repo)
'''),
    cell('markdown', '## Outputs\n\n`/kaggle/working/secap-evaluation/` contains `comprehensive_metrics.json`, a metrics file for every model/dataset pairing, identity-level retrieval ranks, and random/worst 30 distinct-ID contact sheets for each primary cross-domain protocol. Save the notebook output or archive this directory as a private Kaggle dataset.'),
]
nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}}, 'cells': cells}
for i, c in enumerate(cells): c['id'] = f'secap-eval-{i}'
(OUT / 'secap_comprehensive_evaluation.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
meta = {'id':'rizwanabdullah05/secap-comprehensive-evaluation','title':'SeCap Comprehensive Evaluation','code_file':'secap_comprehensive_evaluation.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':['rizwanabdullah05/person-reid-cargo-g2a-lagper','rizwanabdullah05/secap-reproduction-artifacts'],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(OUT / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
print('Built', OUT)
