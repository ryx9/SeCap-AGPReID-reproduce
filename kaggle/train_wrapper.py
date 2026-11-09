"""Kaggle checkpoint/lifecycle adapter; the author's model and losses stay untouched."""
import json
from functools import partial
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from fastreid.engine import DefaultTrainer, default_argument_parser, launch, hooks
from fastreid.utils import comm
from tools.train_net import setup
from kaggle.g2a_adapter import G2AVReID, TrackletEvaluator


class SessionBudgetReached(Exception):
    pass


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


class EpochCheckpoint(hooks.HookBase):
    def before_train(self):
        self.best = -math.inf
        self.best_epoch = None
        path = Path(self.trainer.cfg.OUTPUT_DIR) / 'progress.json'
        if path.exists():
            previous = json.loads(path.read_text())
            self.best = previous.get('best_metric')
            self.best_epoch = previous.get('best_epoch')
            if self.best is None:
                self.best = -math.inf

    def before_epoch(self):
        self.epoch_started = time.time()
        self.epoch_losses = {}
        self.epoch_steps = 0

    def after_step(self):
        self.epoch_steps += 1
        for key, (value, _) in self.trainer.storage.latest().items():
            if 'loss' in key and math.isfinite(float(value)):
                self.epoch_losses[key] = self.epoch_losses.get(key, 0.) + float(value)

    def after_epoch(self):
        t = self.trainer
        rng = dict(python=random.getstate(), numpy=np.random.get_state(),
                   torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())
        states = comm.gather(rng, dst=0)
        epoch_seconds = time.time() - getattr(self, 'epoch_started', time.time())
        stop = time.time() + max(60, epoch_seconds * 1.25) >= float(os.environ.get('SECAP_DEADLINE', 'inf'))
        stop = any(comm.all_gather(stop))
        if comm.is_main_process():
            out = Path(t.cfg.OUTPUT_DIR)
            results = getattr(t, '_last_eval_results', {})
            first = t.cfg.DATASETS.TESTS[0]
            metric = results.get(first, results).get('metric', None)
            evaluated = (t.epoch + 1) % t.cfg.TEST.EVAL_PERIOD == 0
            improved = evaluated and metric is not None and math.isfinite(metric) and metric > self.best
            if improved:
                self.best = metric
                self.best_epoch = t.epoch + 1
            state = dict(epoch=t.epoch, iteration=t.iter, best_metric=self.best, best_epoch=self.best_epoch,
                         rng_by_rank=states)
            # Checkpointer includes model, optimizer, LR schedulers, and registered AMP scaler.
            # Write a new immutable epoch file then atomically publish its aliases.
            payload = {'model': t.checkpointer.model.state_dict(), **state}
            payload.update({k: v.state_dict() for k, v in t.checkpointer.checkpointables.items()})
            torch.save(payload, out / '_epoch_pending.pth')
            epoch_file = out / f'model_epoch_{t.epoch + 1:04d}.pth'
            (out / '_epoch_pending.pth').replace(epoch_file)
            for name in ['model_last.pth'] + (['model_best.pth'] if improved else []):
                tmp = out / (name + '.tmp')
                tmp.unlink(missing_ok=True)
                os.link(epoch_file, tmp)
                tmp.replace(out / name)
            if t.epoch + 1 == t.max_epoch:
                tmp = out / 'model_final.pth.tmp'
                tmp.unlink(missing_ok=True)
                os.link(epoch_file, tmp)
                tmp.replace(out / 'model_final.pth')
            marker = out / 'last_checkpoint.tmp'
            marker.write_text('model_last.pth')
            marker.replace(out / 'last_checkpoint')
            metrics = {k: float(v[0]) for k, v in t.storage.latest().items()
                       if math.isfinite(float(v[0]))}
            metrics.update({'epoch_mean/' + k: v / self.epoch_steps for k, v in getattr(self, 'epoch_losses', {}).items() if self.epoch_steps})
            if improved:
                atomic_json(out / 'best-evaluation.json', dict(epoch=t.epoch + 1, results=results))
            record = dict(epoch=t.epoch + 1, iteration=t.iter, metrics=metrics,
                          evaluation=results if evaluated else {}, time=time.time())
            with (out / 'epochs.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            atomic_json(out / 'progress.json', dict(epoch=t.epoch + 1, max_epoch=t.max_epoch,
                        best_metric=self.best if math.isfinite(self.best) else None, best_epoch=self.best_epoch,
                        complete=t.epoch + 1 == t.max_epoch, budget_stop=stop,
                        checkpoint='model_last.pth', time=time.time()))
            # Keep all epoch metrics, latest two epoch files, best and last full checkpoints.
            for old in sorted(out.glob('model_epoch_*.pth'))[:-2]:
                old.unlink()
            print('SECAP_EPOCH ' + json.dumps(record), flush=True)
        comm.synchronize()
        if stop and t.epoch + 1 < t.max_epoch:
            raise SessionBudgetReached()


def preserve_gradient_buffers(optimizer):
    # PyTorch 2.x changed zero_grad's default to set_to_none=True. FastReID's
    # ContiguousParams requires its preallocated gradient views to stay alive.
    optimizer.zero_grad = partial(optimizer.zero_grad, set_to_none=False)


class KaggleTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        preserve_gradient_buffers(self.optimizer)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        if dataset_name == 'G2AVReID':
            loader, _ = cls.build_test_loader(cfg, dataset_name)
            dataset = G2AVReID(root=os.environ['FASTREID_DATASETS'])
            return loader, TrackletEvaluator(cfg, dataset, output_dir)
        return super().build_evaluator(cfg, dataset_name, output_dir)

    def build_hooks(self):
        original = super().build_hooks()
        # Default checkpoint hook only exists on rank zero. Insert on EVERY rank
        # immediately after evaluation so RNG gathering and barriers are collective.
        ret = []
        for h in original:
            if isinstance(h, hooks.PeriodicCheckpointer):
                continue
            ret.append(h)
            if isinstance(h, hooks.EvalHook):
                ret.append(EpochCheckpoint())
        return ret

    def resume_or_load(self, resume=True):
        if self.grad_scaler is not None:
            self.checkpointer.checkpointables['grad_scaler'] = self.grad_scaler
        super().resume_or_load(resume)
        if resume and self.checkpointer.has_checkpoint():
            state = torch.load(self.checkpointer.get_checkpoint_file(), map_location='cpu')
            states = state.get('rng_by_rank')
            if states:
                rng = states[comm.get_rank()]
                random.setstate(rng['python'])
                np.random.set_state(rng['numpy'])
                torch.set_rng_state(rng['torch'])
                torch.cuda.set_rng_state(rng['cuda'])


def main(args):
    cfg = setup(args)
    trainer = KaggleTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    try:
        trainer.train()
    except SessionBudgetReached:
        print('SECAP_BUDGET_STOP: completed epoch checkpoint saved', flush=True)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    launch(main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
