"""Tiny synthetic metadata fixture tests the official protocol parser and video pooling."""
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.io import savemat
import torch
from kaggle.g2a_adapter import G2AVReID, TrackletEvaluator
from fastreid.evaluation import ReidEvaluator

root = Path('/tmp/secap-g2a-test')
if root.exists():
    shutil.rmtree(root)
folder = root / 'G2A-VReID'
info = folder / 'info'
info.mkdir(parents=True)
for subset, pid in [('train', 1), ('test', 2)]:
    names = [f'{pid:04d}C{cam}T0001F{frame:03d}.jpg' for cam in [1, 2] for frame in [1, 2]]
    for name in names:
        path = folder / f'bbox_{subset}' / name[:4] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (info / f'{subset}_name.txt').write_text('\n'.join(names) + '\n')
    savemat(info / f'tracks_{subset}_info.mat', {f'track_{subset}_info':np.array([[1,2,pid,1],[3,4,pid,2]])})
savemat(info / 'query_IDX.mat', {'query_IDX':np.array([[2]])})
data = G2AVReID(root=str(root))
assert len(data.train) == 4 and len(data.query) == 2 and len(data.gallery) == 2
assert {x[3] for x in data.query} == {'Ground'}
assert {x[3] for x in data.gallery} == {'Aerial'}
assert data.view_mapping == {2:'Ground',1:'Aerial'}
e = TrackletEvaluator(NS(), data)
e.reset()
paths = [x[0] for x in data.query + data.gallery]
e.process({'img_paths':paths[:3]}, torch.tensor([[1.,3.],[3.,5.],[10.,20.]]))
e.process({'img_paths':paths[3:]}, torch.tensor([[20.,40.]]))
original = ReidEvaluator.evaluate
try:
    ReidEvaluator.evaluate = lambda self: self._predictions[0]['feats']
    pooled = e.evaluate()
finally:
    ReidEvaluator.evaluate = original
assert torch.equal(pooled, torch.tensor([[2.,4.],[15.,30.]]))
# Missing frames must fail instead of silently changing the video protocol.
Path(paths[0]).unlink()
try:
    G2AVReID(root=str(root))
except FileNotFoundError:
    pass
else:
    raise AssertionError('Missing tracklet frame was accepted')
shutil.rmtree(root)
print('SECAP_G2A_TEST_PASSED: official query indices, identity isolation, view mapping, frame completeness and video mean pooling')
