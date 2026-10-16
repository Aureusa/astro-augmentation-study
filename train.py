"""Single entrypoint for every GZD5 augmentation experiment.

    python train.py --experiment walmsley_transforms --seed 42 --train-fraction 0.05
    python train.py --list                                   # registered experiments (experiments.py)

Protocol (Walmsley et al. 2022 style): AdamW, constant LR after a short linear warmup, early stopping on
val_total_loss with a patience counted in evaluations. `--max-epochs` is only a safety cap. Evaluation runs
every epoch, but at least every `--min-eval-iters` iterations, so tiny training subsets do not spend all their
time on the (full) validation set. After training, the best checkpoint is evaluated once on the test split.

Output: <runs-root>/frac_<f>/<experiment>_<seed>/ with log.log, config.json, train_subset.npy,
ckpt_best_val_total_loss.pt, the latest resumable checkpoint (older ones are deleted), and DONE.json once
finished. Re-running the same command resumes an unfinished run and skips a finished one.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from data.dataset import DECaLSDataset
from experiments import EXPERIMENTS, get_experiment
from model.efficientnet import get_model
from task.gdz_5_schema import GZD5_QUESTIONS
from task import gz_decals, gz_decals_augmix  # noqa: F401  (register gzd5_task, gzd5_augmix_task)

import transforms  # noqa: F401  (registers the astro_augment transforms and batch transforms)

from optastra.data import build_dataloader
from optastra.optim import Optimizer, Scheduler
from optastra.tasks import Task
from optastra.training import Trainer
from optastra.training.checkpointer import Checkpointer
from optastra.training.hooks.batch_transform import BatchTransformHook
from optastra.training.hooks.checkpoint import BestCheckpointHook, CheckpointHook
from optastra.training.hooks.common_metrics_printer import CommonMetricPrinterHook
from optastra.training.hooks.early_stopping import EarlyStoppingHook
from optastra.training.hooks.eval import EvalHook
from optastra.training.hooks.resume import ResumeHook
from optastra.training.hooks.scheduler import SchedulerHook
from optastra.training.hooks.writer import JSONWriterHook
from optastra.training.logging_config import setup_logging
from optastra.transforms import BatchTransform, Compose, Transform

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Galaxy Zoo DECaLS data (parquets, PNG zips and the split parquets from make_splits.py); see README.md.
DATA_DIR = Path(os.environ.get("GZD5_DATA_DIR", Path.home() / "project_data" / "aa_gz_decals"))
SPLITS = {s: DATA_DIR / f"decals_split_{s}.parquet" for s in ("train", "val", "test")}
METRIC = "val_total_loss"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", help="name from experiments.py (see --list)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-fraction", type=float, default=1.0, help="fraction of the training split to use")
    p.add_argument("--subset-seed", type=int, default=0,
                   help="seed of the training-subset draw; fixed, so subsets are nested across fractions and "
                        "identical across training seeds")
    p.add_argument("--runs-root", default="runs/sweep")
    p.add_argument("--output-dir", default=None, help="overrides <runs-root>/frac_<f>/<experiment>_<seed>")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--schedule", choices=["constant", "cosine"], default="constant",
                   help="constant: LR flat after warmup (early stopping decides the length). "
                        "cosine: the old protocol, cosine to 0 over --cosine-epochs")
    p.add_argument("--cosine-epochs", type=int, default=100)
    p.add_argument("--max-epochs", type=int, default=1000, help="safety cap; runs are meant to end by early stopping")
    p.add_argument("--patience", type=int, default=10, help="early-stopping patience, in evaluations")
    p.add_argument("--min-eval-iters", type=int, default=200,
                   help="evaluate every epoch, but no more often than every this many iterations")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16",
                   help="bf16 matches the old runs (autocast used to live in the task)")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--no-test", action="store_true", help="skip the final test-split evaluation")
    p.add_argument("--val-size", type=int, default=None,
                   help="model selection on a fixed random subset of N val images (same for every run); "
                        "None = the full val split. Final numbers come from the full test split either way")
    p.add_argument("--limit-eval", type=int, default=None, help="debug only: use the first N val/test images")
    p.add_argument("--list", action="store_true", help="list registered experiments and exit")
    args = p.parse_args(argv)
    if not args.list and not args.experiment:
        p.error("--experiment is required (or use --list)")
    if not 0 < args.train_fraction <= 1:
        p.error("--train-fraction must be in (0, 1]")
    return args


def default_output_dir(args) -> Path:
    return Path(args.runs_root) / f"frac_{args.train_fraction:g}" / f"{args.experiment}_{args.seed}"


class LatestCheckpointHook(CheckpointHook):
    """CheckpointHook that keeps only the newest periodic checkpoint (enough to resume), so a run does not
    leave one 50 MB file per epoch behind. The best checkpoint is written separately by BestCheckpointHook."""

    _PERIODIC = re.compile(r"^ckpt_(\d+)\.pt$")

    def after_step(self, state) -> None:
        if state.iter == 0 or state.iter % self.save_every != 0:
            return
        keep = os.path.basename(self.checkpointer.save(state, self.checkpointer.periodic_name(state.iter)))
        for f in os.listdir(self.checkpointer.output_dir):
            if self._PERIODIC.match(f) and f != keep:
                os.remove(os.path.join(self.checkpointer.output_dir, f))


def _git_info(path: Path) -> dict:
    def git(*cmd):
        try:
            return subprocess.run(["git", "-C", str(path), *cmd], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def build_pipeline(names) -> Compose:
    return Compose([Transform.create("to_float"), *[Transform.create(n) for n in names]])


def make_loader(dataset, task, batch_size, shuffle, num_workers):
    return build_dataloader(dataset, task=task, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None,
                            pin_memory=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list:
        for name, exp in sorted(EXPERIMENTS.items()):
            print(f"{name:32s} train={'+'.join(exp.train):45s} batch={'+'.join(exp.batch) or '-'}")
        return 0

    exp = get_experiment(args.experiment)
    out = Path(args.output_dir) if args.output_dir else default_output_dir(args)
    if (out / "DONE.json").exists():
        print(f"[skip] {out} is already finished (DONE.json)")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out), filename="log.log")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- data: a fixed, nested random subset of the training split
    train_full = DECaLSDataset(parquet_path=SPLITS["train"], data_root=DATA_DIR, transform=build_pipeline(exp.train))
    n_full = len(train_full)
    n_train = max(args.batch_size, int(round(args.train_fraction * n_full)))
    order = np.random.default_rng(args.subset_seed).permutation(n_full)
    subset_idx = np.sort(order[:n_train])
    np.save(out / "train_subset.npy", subset_idx)
    train_ds = Subset(train_full, subset_idx.tolist()) if n_train < n_full else train_full
    val_ds = DECaLSDataset(parquet_path=SPLITS["val"], data_root=DATA_DIR, transform=build_pipeline(exp.val))
    if args.val_size and args.val_size < len(val_ds):
        val_idx = np.sort(np.random.default_rng(args.subset_seed + 1).permutation(len(val_ds))[:args.val_size])
        val_ds = Subset(val_ds, val_idx.tolist())
    if args.limit_eval:
        val_ds = Subset(val_ds, list(range(min(args.limit_eval, len(val_ds)))))

    task = Task.create(exp.task)
    train_loader = make_loader(train_ds, task, args.batch_size, True, args.num_workers)
    eval_workers = max(1, args.num_workers // 2)      # the val loader is idle between evaluations
    val_loader = make_loader(val_ds, task, args.batch_size, False, eval_workers)
    iters_per_epoch = len(train_loader)
    eval_period = max(iters_per_epoch, args.min_eval_iters)
    max_iter = args.max_epochs * iters_per_epoch

    # ---- model, optimiser, schedule
    num_answers = sum(len(q.answers) for q in GZD5_QUESTIONS)
    model = get_model(in_channels=exp.in_channels, num_classes=num_answers, dropout=0.0)
    optimizer = Optimizer.create("adamw", model, lr=args.lr)
    if args.schedule == "constant":
        warm, start = max(args.warmup_steps, 1), 0.01
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: start + (1 - start) * min(1.0, s / warm))
    else:
        total = args.cosine_epochs * iters_per_epoch
        scheduler = Scheduler.create("warmup_cosine", optimizer=optimizer, warmup_steps=min(args.warmup_steps, total),
                                     total_steps=total)

    config = {
        "args": vars(args), "experiment": exp.describe(), "output_dir": str(out),
        "train_size": n_train, "train_size_full": n_full, "val_size": len(val_ds),
        "iters_per_epoch": iters_per_epoch, "eval_period_iters": eval_period, "max_iter": max_iter,
        "optastra_git": _git_info(Path(__file__).resolve().parent.parent / "optastra"),
        "python": sys.version.split()[0], "torch": torch.__version__, "host": os.uname().nodename,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    logger.info(f"config: {json.dumps(config)}")
    logger.info(f"experiment {exp.name}: train={exp.describe()['train']} val={exp.describe()['val']} batch={list(exp.batch)}")
    logger.info(f"train subset {n_train}/{n_full} ({args.train_fraction:g}), {iters_per_epoch} iters/epoch, "
                f"eval every {eval_period} iters, patience {args.patience} evals, cap {max_iter} iters")

    # ---- trainer and hooks
    trainer = Trainer(model=model, task=task, optimizer=optimizer, precision=args.precision)
    checkpointer = Checkpointer(str(out))
    early_stopping = EarlyStoppingHook(metric=METRIC, patience=args.patience)
    hooks = [
        ResumeHook(checkpointer),
        LatestCheckpointHook(checkpointer, save_every=eval_period),
        BestCheckpointHook(checkpointer, metric=METRIC, mode="min"),
        CommonMetricPrinterHook(log_every=args.log_every),
        JSONWriterHook(output_dir=str(out / "logs"), log_every=args.log_every),
        SchedulerHook(scheduler),
        EvalHook(eval_period, eval_fn=lambda: trainer.evaluate(val_loader)),
        early_stopping,
    ]
    if exp.batch:
        hooks.append(BatchTransformHook([BatchTransform.create(n) for n in exp.batch]))
    trainer.register_hooks(hooks)

    t0 = time.time()
    trainer.train(train_loader, max_iter=max_iter)
    stopped_early = bool(trainer.state.should_stop)
    done = {"stopped_early": stopped_early, "last_iter": int(trainer.state.iter),
            "best_" + METRIC: float(early_stopping.tracker.best), "train_seconds": round(time.time() - t0, 1)}
    if not stopped_early:
        logger.warning(f"run hit the {max_iter}-iteration cap without early stopping; raise --max-epochs")

    # ---- test split, best checkpoint
    best_path = Path(checkpointer.path(checkpointer.best_name(METRIC)))
    if not args.no_test and best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu")
        Checkpointer.load_model_state_dict(model, ckpt["model"])
        test_ds = DECaLSDataset(parquet_path=SPLITS["test"], data_root=DATA_DIR, transform=build_pipeline(exp.val))
        if args.limit_eval:
            test_ds = Subset(test_ds, list(range(min(args.limit_eval, len(test_ds)))))
        tester = Trainer(model=model, task=task, optimizer=optimizer, precision=args.precision)
        test_metrics = tester.evaluate(make_loader(test_ds, task, args.batch_size, False, eval_workers), prefix="test")
        done["best_iter"] = int(ckpt["iter"])
        done["test"] = {k: float(v) for k, v in test_metrics.items()}
        logger.info(f"test metrics (best checkpoint, iter {ckpt['iter']}): "
                    + "  ".join(f"test_{k}={v:.4f}" for k, v in test_metrics.items()))
    done["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "DONE.json").write_text(json.dumps(done, indent=2))
    logger.info(f"DONE: {json.dumps(done)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
