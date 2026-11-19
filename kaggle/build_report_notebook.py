"""Build visual report notebook for completed SeCap evaluation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'kaggle/report_notebook'
OUT.mkdir(exist_ok=True)

def cell(kind, source):
    item = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code': item.update(execution_count=None, outputs=[])
    return item

cells = [
cell('markdown', '# SeCap comprehensive evaluation report\n\nMetrics, identity difficulty, and retrieval-image audits for both best checkpoints.'),
cell('code', '''from pathlib import Path
import json, zipfile, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Markdown
for archive in Path('/kaggle/input').rglob('secap-evaluation.zip'):
    with zipfile.ZipFile(archive) as z: z.extractall('/kaggle/working/secap-report-input')
files=list(Path('/kaggle/input').rglob('comprehensive_metrics.json')) + list(Path('/kaggle/working/secap-report-input').rglob('comprehensive_metrics.json'))
assert len(files)==1, files
RESULTS=files[0].parent
METRICS=json.loads(files[0].read_text())
print('Results:', RESULTS)'''),
cell('markdown', '## Complete metrics'),
cell('code', '''rows=[]
for pair, report in METRICS.items():
    for protocol, score in report['protocols'].items():
        rows.append({'pair':pair,'source_model':report['source_model'],'target_dataset':report['target_data'],'protocol':protocol,**score})
table=pd.DataFrame(rows).sort_values(['source_model','target_dataset','protocol'])
display(table.style.format({k:'{:.2f}' for k in ['Rank-1','Rank-5','Rank-10','mAP','mINP','metric']}))
table.to_csv('/kaggle/working/secap_evaluation_metrics.csv',index=False)'''),
cell('markdown', '## Rank-1 and mAP'),
cell('code', '''fig,axes=plt.subplots(1,2,figsize=(18,6))
labels=[f'{r.source_model} → {r.target_dataset}\\n{r.protocol}' for r in table.itertuples()]
for ax,metric in zip(axes,['Rank-1','mAP']):
    bars=ax.bar(range(len(table)),table[metric],color=['#1976d2' if a==b else '#ef6c00' for a,b in zip(table.source_model,table.target_dataset)])
    ax.set(title=metric+' (%)',ylabel='%'); ax.set_xticks(range(len(table)),labels,rotation=90); ax.grid(axis='y',alpha=.25)
    for bar,value in zip(bars,table[metric]): ax.text(bar.get_x()+bar.get_width()/2,value,f'{value:.1f}',ha='center',va='bottom',fontsize=7,rotation=90)
fig.tight_layout(); fig.savefig('/kaggle/working/secap_metric_comparison.png',dpi=180,bbox_inches='tight'); plt.show()'''),
cell('markdown', '## Identities with the greatest retrieval difficulty'),
cell('code', '''difficulty=[]
for pair_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
    for path in pair_dir.glob('*_identity_failures.json'):
        data=json.loads(path.read_text())
        if data:
            x=pd.DataFrame(data).groupby('pid').agg(worst_rank=('rank','max'),mean_rank=('rank','mean'),mean_ap=('ap','mean'),queries=('pid','size')).reset_index()
            x['pair']=pair_dir.name; x['protocol']=path.name.replace('_identity_failures.json',''); difficulty.append(x)
hardest=pd.concat(difficulty).sort_values(['worst_rank','mean_rank'],ascending=False).head(30)
display(hardest.style.format({'mean_rank':'{:.2f}','mean_ap':'{:.4f}'})); hardest.to_csv('/kaggle/working/secap_hardest_identities.csv',index=False)'''),
cell('markdown', '## Random-30 and worst-30 retrieval previews'),
cell('code', '''for pair_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
    sheets=sorted(pair_dir.glob('*_random_30.png'))+sorted(pair_dir.glob('*_worst_30.png'))
    if sheets: display(Markdown('### '+pair_dir.name))
    for path in sheets:
        display(Markdown('**'+path.stem.replace('_',' ')+'**'))
        with Image.open(path) as image: display(image.copy())'''),
]
for i,item in enumerate(cells): item['id']=f'secap-report-{i}'
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
(OUT/'secap_evaluation_report.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
meta={'id':'rizwanabdullah05/secap-evaluation-visual-report','title':'SeCap Evaluation Visual Report','code_file':'secap_evaluation_report.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_internet':False,'machine_shape':'standard','dataset_sources':['rizwanabdullah05/secap-comprehensive-evaluation-results'],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(OUT/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Built',OUT)
