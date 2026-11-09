"""Run in the Kaggle training environment before committing long GPU training."""
import json
import contextlib
import io
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from fastreid.utils.checkpoint import Checkpointer
from kaggle.train_wrapper import EpochCheckpoint, preserve_gradient_buffers
from fastreid.utils.params import ContiguousParams

out = Path('/tmp/secap-checkpoint-test')
if out.exists():
    shutil.rmtree(out)
out.mkdir()
model = torch.nn.Linear(2, 2).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.9)
scaler = torch.cuda.amp.GradScaler()
check = Checkpointer(model, str(out), optimizer=optimizer, lr_sched=scheduler, grad_scaler=scaler)
trainer = NS(cfg=NS(OUTPUT_DIR=str(out), DATASETS=NS(TESTS=['fixture']), TEST=NS(EVAL_PERIOD=1)),
             checkpointer=check, max_epoch=4)
trainer.storage = NS(latest=lambda: {'total_loss': (1.0, trainer.iter)})
hook = EpochCheckpoint()
hook.trainer = trainer
hook.before_train()
for epoch, metric in enumerate([50., 40., 60., 55.]):
    optimizer.zero_grad()
    scaler.scale(model(torch.ones(2, 2, device='cuda')).sum()).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    trainer.epoch, trainer.iter = epoch, (epoch + 1) * 10
    trainer._last_eval_results = {'metric': metric}
    hook.before_epoch()
    hook.after_step()
    with contextlib.redirect_stdout(io.StringIO()):
        hook.after_epoch()

last = torch.load(out / 'model_last.pth', map_location='cpu')
best = torch.load(out / 'model_best.pth', map_location='cpu')
assert last['epoch'] == 3 and best['epoch'] == 2
assert last['best_metric'] == 60.
assert len(list(out.glob('model_epoch_*.pth'))) == 2
assert len((out / 'epochs.jsonl').read_text().splitlines()) == 4
assert (out / 'last_checkpoint').read_text() == 'model_last.pth'
assert json.loads((out / 'progress.json').read_text())['complete']
assert all(k in last for k in ['model', 'optimizer', 'lr_sched', 'grad_scaler', 'rng_by_rank'])
# Exercise the actual author resume code path with stateful optimizer and schedule.
other = torch.nn.Linear(2, 2).cuda()
opt2 = torch.optim.SGD(other.parameters(), lr=.7, momentum=.9)
sched2 = torch.optim.lr_scheduler.StepLR(opt2, 1, gamma=.9)
scale2 = torch.cuda.amp.GradScaler()
resumed = Checkpointer(other, str(out), optimizer=opt2, lr_sched=sched2, grad_scaler=scale2)
metadata = resumed.resume_or_load('', resume=True)
assert metadata['epoch'] + 1 == 4
for p, q in zip(model.parameters(), other.parameters()):
    assert torch.equal(p, q)
assert opt2.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
assert sched2.state_dict() == scheduler.state_dict()
assert scale2.state_dict() == scaler.state_dict()
hook2 = EpochCheckpoint()
hook2.trainer = trainer
hook2.before_train()
assert hook2.best == 60.
# Regression: exercise the author's flattened optimizer buffers with AMP.
flat_model = torch.nn.Linear(2, 2).cuda()
flat_params = ContiguousParams([dict(params=[p], lr=.1, weight_decay=0., freeze_status='normal')
                                for p in flat_model.parameters()])
flat_optimizer = torch.optim.SGD(flat_params.contiguous(), lr=.1, momentum=.9)
preserve_gradient_buffers(flat_optimizer)
flat_scaler = torch.cuda.amp.GradScaler()
before = flat_model.weight.detach().clone()
for _ in range(2):
    flat_optimizer.zero_grad()
    flat_scaler.scale(flat_model(torch.ones(2, 2, device='cuda')).sum()).backward()
    flat_scaler.step(flat_optimizer)
    flat_scaler.update()
    flat_params.assert_buffer_is_valid()
assert not torch.equal(before, flat_model.weight)
print('SECAP_CONTIGUOUS_AMP_TEST_PASSED: real gradients update flattened parameters across repeated steps')
print('SECAP_CHECKPOINT_TEST_PASSED: best/last, retention, optimizer, scheduler, scaler, RNG and epoch resume')
shutil.rmtree(out)
