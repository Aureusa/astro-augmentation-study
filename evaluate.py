"""Evaluate the best checkpoint of the old (pre-sweep) runs on the test split.

    python evaluate.py                       # every runs/<experiment>_<seed>[_tag] directory
    python evaluate.py --only walmsley_transforms_42 baseline_A_42 --splits val test

For each run directory:
  1. The best eval is read from log.log (the 4-decimal `optastra.eval ... val_total_loss=` lines). Old runs
     saved a checkpoint every epoch but not at the final post-training eval, so the best eval *that has a
     checkpoint* is used; TEST.json records whether that is the overall best.
  2. The checkpoint is loaded into a fresh model with the current optastra (strict state_dict load).
  3. It is evaluated on the requested splits with the experiment's eval transforms from experiments.py
     (center crop; resize for baseline_A, grayscale + crop for walmsley_transforms_gray). With `val`
     included, the recomputed val_total_loss is compared with the logged one: a match verifies both the
     checkpoint loading and the eval transforms.

Writes <run>/TEST.json and results/tables/old_runs_test.csv.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from data.dataset import DECaLSDataset
from experiments import EXPERIMENTS
from model.efficientnet import get_model
from task.gdz_5_schema import GZD5_QUESTIONS
from task.gz_decals import GZD5Task
from train import DATA_DIR, SPLITS, build_pipeline

import transforms  # noqa: F401

from optastra.data import build_dataloader
from optastra.optim import Optimizer
from optastra.training import Trainer
from optastra.training.checkpointer import Checkpointer

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

RUN_DIR_RE = re.compile(r"^(?P<experiment>.+)_(?P<seed>\d+)(?P<tag>_[A-Za-z]+)?$")
EVAL_RE = re.compile(r"optastra\.eval INFO\]: eval at iter (?P<iter>\d+)/\d+\s.*?val_total_loss=(?P<loss>[-+0-9.eE]+)")
TOL = 0.01          # |recomputed - logged| val_total_loss that counts as a match


def eval_transforms(experiment: str):
    """(val transform names, in_channels). RandomResizedCrop runs are no longer registered; they were
    validated on a plain center crop like every other run."""
    if experiment in EXPERIMENTS:
        exp = EXPERIMENTS[experiment]
        return exp.val, exp.in_channels
    return ("crop",), 3


def best_eval_with_checkpoint(run_dir: Path):
    evals = {}
    for line in open(run_dir / "log.log", encoding="utf-8", errors="replace"):
        m = EVAL_RE.search(line)
        if m:
            evals[int(m.group("iter"))] = float(m.group("loss"))      # resumed runs: last occurrence wins
    if not evals:
        return None
    overall = min(evals, key=evals.get)
    with_ckpt = {it: v for it, v in evals.items() if (run_dir / f"ckpt_{it}.pt").exists()}
    if not with_ckpt:
        return {"overall_best_iter": overall, "overall_best_loss": evals[overall], "iter": None}
    it = min(with_ckpt, key=with_ckpt.get)
    return {"overall_best_iter": overall, "overall_best_loss": evals[overall], "iter": it, "logged_loss": evals[it],
            "is_overall_best": it == overall}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--only", nargs="*", help="run directory names to evaluate (default: all)")
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    p.add_argument("--extra-val-transforms", nargs="*", default=[],
                   help="also evaluate val with these transforms, e.g. `crop` for baseline_A, to identify the "
                        "transform the run was validated with")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--precision", default="bf16")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--table", default="results/tables/old_runs_test.csv")
    args = p.parse_args(argv)

    root = Path(args.runs_root)
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and RUN_DIR_RE.match(d.name) and (d / "log.log").exists())
    if args.only:
        dirs = [d for d in dirs if d.name in set(args.only)]
    task = GZD5Task()
    num_answers = sum(len(q.answers) for q in GZD5_QUESTIONS)
    datasets = {}

    def loader(split, names):
        # Datasets are cached; loaders are not, and their workers are not persistent, so worker processes
        # only exist during one evaluation (cached persistent loaders keep every pipeline's workers alive).
        key = (split, tuple(names))
        if key not in datasets:
            datasets[key] = DECaLSDataset(parquet_path=SPLITS[split], data_root=DATA_DIR, transform=build_pipeline(names))
        return build_dataloader(datasets[key], task=task, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, persistent_workers=False, pin_memory=True)

    rows = []
    for d in dirs:
        out_file = d / "TEST.json"
        if out_file.exists() and not args.overwrite:
            rows.append(json.loads(out_file.read_text()))
            continue
        experiment, seed = RUN_DIR_RE.match(d.name).group("experiment"), RUN_DIR_RE.match(d.name).group("seed")
        val_names, in_channels = eval_transforms(experiment)
        rec = {"run": d.name, "experiment": experiment, "seed": seed, "eval_transforms": list(val_names)}
        best = best_eval_with_checkpoint(d)
        if best is None or best["iter"] is None:
            rec["error"] = "no eval lines" if best is None else "no checkpoint for any eval"
            print(f"[{d.name}] {rec['error']}", flush=True)
            rows.append(rec)
            continue
        rec.update(best)
        t0 = time.time()
        try:
            model = get_model(in_channels=in_channels, num_classes=num_answers, dropout=0.0)
            ckpt = torch.load(d / f"ckpt_{best['iter']}.pt", map_location="cpu")
            Checkpointer.load_model_state_dict(model, ckpt["model"])        # strict: fails on any key mismatch
            trainer = Trainer(model=model, task=task, optimizer=Optimizer.create("adamw", model, lr=1e-4),
                              precision=args.precision)
            for split in args.splits:
                rec[split] = {k: float(v) for k, v in trainer.evaluate(loader(split, val_names), prefix=split).items()}
            for names in args.extra_val_transforms:
                alt = [n for n in names.split("+")]
                rec[f"val[{names}]"] = {k: float(v) for k, v in
                                        trainer.evaluate(loader("val", alt), prefix="val").items()}
        except Exception as err:                                            # keep going; report at the end
            rec["error"] = f"{type(err).__name__}: {err}"
            print(f"[{d.name}] ERROR {rec['error']}", flush=True)
            rows.append(rec)
            continue
        if "val" in rec:
            rec["val_matches_log"] = abs(rec["val"]["total_loss"] - rec["logged_loss"]) < TOL
        rec["eval_seconds"] = round(time.time() - t0, 1)
        out_file.write_text(json.dumps(rec, indent=2))
        msg = f"[{d.name}] ckpt {best['iter']}" + ("" if best["is_overall_best"] else f" (overall best {best['overall_best_iter']} has no ckpt)")
        if "val" in rec:
            msg += f"  val {rec['val']['total_loss']:.4f} vs logged {rec['logged_loss']:.4f} -> {'OK' if rec['val_matches_log'] else 'MISMATCH'}"
        if "test" in rec:
            msg += f"  test {rec['test']['total_loss']:.4f}"
        print(msg + f"  ({rec['eval_seconds']:.0f}s)", flush=True)
        rows.append(rec)

    flat = []
    for r in rows:
        f = {k: v for k, v in r.items() if not isinstance(v, dict)}
        for split in ("val", "test"):
            for k, v in (r.get(split) or {}).items():
                f[f"{split}_{k}"] = v
        flat.append(f)
    table = pd.DataFrame(flat)
    Path(args.table).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.table, index=False)
    n_err = int(table["error"].notna().sum()) if "error" in table else 0
    n_mis = int((table.get("val_matches_log") == False).sum()) if "val_matches_log" in table else 0  # noqa: E712
    print(f"wrote {args.table}: {len(table)} runs, {n_err} errors, {n_mis} val mismatches")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
