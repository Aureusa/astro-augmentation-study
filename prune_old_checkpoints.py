"""Delete the per-epoch checkpoints of the old runs, keeping the best one (the one evaluate.py used).

    python3 prune_old_checkpoints.py              # dry run: list what would be deleted and the space freed
    python3 prune_old_checkpoints.py --delete     # actually delete
    python3 prune_old_checkpoints.py --delete --keep-last   # also keep each run's last checkpoint

Only touches runs/<experiment>_<seed>[_tag] directories that have a TEST.json without an error, and keeps
ckpt_<iter>.pt for the iter recorded there. Runs without TEST.json (not evaluated yet, or failed) are left
alone, as is everything under runs/sweep/. Needs only the standard library, so it runs on the login node.
"""
import argparse
import json
import re
from pathlib import Path

RUN_DIR_RE = re.compile(r"^.+_\d+(_[A-Za-z]+)?$")
CKPT_RE = re.compile(r"^ckpt_(\d+)\.pt(\.tmp)?$")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--delete", action="store_true", help="delete (default is a dry run)")
    p.add_argument("--keep-last", action="store_true", help="also keep the highest-iteration checkpoint")
    p.add_argument("--require-val-match", action="store_true",
                   help="only prune runs whose recomputed val loss matched the log (val_matches_log)")
    args = p.parse_args()

    freed, n_files, pruned, skipped = 0, 0, 0, []
    for d in sorted(Path(args.runs_root).iterdir()):
        if not d.is_dir() or not RUN_DIR_RE.match(d.name):
            continue
        ckpts = {f: CKPT_RE.match(f.name) for f in d.iterdir() if CKPT_RE.match(f.name)}
        if not ckpts:
            continue
        test = d / "TEST.json"
        if not test.exists():
            skipped.append((d.name, "no TEST.json"))
            continue
        rec = json.loads(test.read_text())
        if rec.get("error") or rec.get("iter") is None:
            skipped.append((d.name, f"TEST.json has an error: {rec.get('error')}"))
            continue
        if args.require_val_match and not rec.get("val_matches_log", False):
            skipped.append((d.name, "val loss did not match the log"))
            continue
        keep = {f"ckpt_{rec['iter']}.pt"}
        if not (d / f"ckpt_{rec['iter']}.pt").exists():
            skipped.append((d.name, f"best checkpoint ckpt_{rec['iter']}.pt is missing"))
            continue
        if args.keep_last:
            finished = [f for f, m in ckpts.items() if not m.group(2)]
            keep.add(max(finished, key=lambda f: int(ckpts[f].group(1))).name)
        to_delete = [f for f in ckpts if f.name not in keep]
        size = sum(f.stat().st_size for f in to_delete)
        if to_delete:
            pruned += 1
            n_files += len(to_delete)
            freed += size
            print(f"{d.name:36s} keep {', '.join(sorted(keep)):28s} delete {len(to_delete):3d} files ({size / 1e9:6.2f} GB)")
            if args.delete:
                for f in to_delete:
                    f.unlink()

    for name, why in skipped:
        print(f"[skip] {name}: {why}")
    verb = "deleted" if args.delete else "would delete"
    print(f"\n{verb} {n_files} checkpoint files in {pruned} runs, {freed / 1e9:.1f} GB; skipped {len(skipped)} runs"
          + ("" if args.delete else "\n(dry run -- pass --delete to actually remove them)"))


if __name__ == "__main__":
    main()
