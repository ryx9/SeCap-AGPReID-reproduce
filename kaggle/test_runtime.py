"""Local standard-library checks: no images, GPUs or model packages are downloaded."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('secap_runtime', Path(__file__).with_name('runtime.py'))
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimeTests(unittest.TestCase):
    def test_dataset_roots_and_split_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input'
            lag = source / 'LAGPeR/LAGPeR'
            cargo = source / 'CARGO/CARGO'
            g2a = source / 'G2A-VReID/G2A-VReID'
            for path in [lag/'bounding_box_train/0001/a.jpg', lag/'query_aerial/0002/b.jpg',
                         cargo/'train/Cam1/c.jpg', cargo/'query/Cam1/d.jpg', cargo/'gallery/Cam2/e.jpg']:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            for split in ['exp3_A2G+.txt','exp4_G2A+.txt','exp5_G2A+G.txt']:
                (lag / split).write_text('query_aerial/0002/b.jpg')
            for folder in ['bbox_train','bbox_test','info']:
                (g2a/folder).mkdir(parents=True, exist_ok=True)
            walked = list(os.walk(source))
            data, art = root/'links', root/'artifacts'
            art.mkdir()
            with patch.object(runtime, 'DATA', data), patch.object(runtime, 'ART', art), patch.object(runtime.os,'walk',return_value=iter(walked)), patch.dict(os.environ):
                runtime.prepare_data()
                report=json.loads((art/'dataset-validation.json').read_text())
                self.assertEqual(report['LAGPeR']['train_images'],1)
                self.assertTrue((data/'CARGO/train').is_symlink())
                self.assertTrue((data/'G2A-VReID').is_symlink())
                self.assertTrue((data/'LAGPeR/exp3_A2G+.txt').read_text().endswith('\n'))

    def test_checkpoint_snapshot_hardlinks_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            art = work/'artifacts'
            out = art/'LAGPeR'
            out.mkdir(parents=True)
            epoch = out/'model_epoch_0001.pth'
            epoch.write_bytes(b'fixture checkpoint')
            os.link(epoch,out/'model_last.pth')
            os.link(epoch,out/'model_best.pth')
            (out/'progress.json').write_text(json.dumps({'epoch':1,'complete':False}))
            settings={'batch_size':64,'stride':[16,16],'eval_period':5}
            (art/'run-settings.json').write_text(json.dumps(settings))
            saved=work/'remote-archive'
            def upload(handle, folder, version_notes):
                import shutil
                shutil.copytree(folder,saved)
            fake=types.SimpleNamespace(dataset_upload=upload,dataset_download=lambda h:str(saved))
            with patch.object(runtime,'WORK',work),patch.object(runtime,'ART',art),patch.dict(sys.modules,{'kagglehub':fake}):
                runtime.snapshot_upload('fixture/private-archive','test')
            self.assertFalse((work/'.secap-upload').exists())
            with tarfile.open(saved/'checkpoints.tar') as bundle:
                members=bundle.getmembers()
                self.assertEqual(sum(m.isfile() for m in members),1)
                self.assertEqual(sum(m.islnk() for m in members),2)
            dest=work/'restored'
            dest.mkdir()
            (dest/'run-settings.json').write_text(json.dumps(settings))
            with patch.object(runtime,'ART',dest),patch.dict(sys.modules,{'kagglehub':fake}):
                runtime.restore('fixture/private-archive')
            self.assertEqual((dest/'LAGPeR/model_last.pth').read_bytes(),b'fixture checkpoint')
            self.assertEqual((dest/'LAGPeR/last_checkpoint').read_text(),'model_last.pth')
            # Kaggle's actual Dataset processing expands .tar files before mounting.
            with tarfile.open(saved/'checkpoints.tar') as bundle:
                bundle.extractall(saved/'checkpoints', filter='data')
            (saved/'checkpoints.tar').unlink()
            mounted = work/'restored-expanded'
            mounted.mkdir()
            (mounted/'run-settings.json').write_text(json.dumps(settings))
            with patch.object(runtime,'ART',mounted),patch.dict(sys.modules,{'kagglehub':fake}):
                runtime.restore('fixture/private-archive')
            self.assertEqual((mounted/'LAGPeR/model_last.pth').read_bytes(),b'fixture checkpoint')


if __name__=='__main__':
    unittest.main()
