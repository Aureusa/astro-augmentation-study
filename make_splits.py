"""Create the fixed train / val / test splits of Galaxy Zoo DECaLS (GZD-5) used by every experiment.

    python make_splits.py                       # writes $GZD5_DATA_DIR/decals_split_{train,val,test}.parquet
    python make_splits.py --check               # only compare with the existing split parquets

Existing split parquets are never replaced unless --overwrite is given.

Galaxies from the GZD-5 volunteer table with at least MIN_VOTES votes on the first question
(smooth-or-featured) are split 80/10/10, stratified by the majority answer to that question, with a fixed
random_state. The result: 199,664 / 24,958 / 24,959 galaxies. Each split parquet holds the full volunteer
rows (identifiers, PNG location and vote counts) of its galaxies.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

MIN_VOTES = 3
SEED = 42
FIRST_QUESTION = ["smooth-or-featured_smooth_fraction", "smooth-or-featured_featured-or-disk_fraction",
                  "smooth-or-featured_artifact_fraction"]


def make_splits(volunteers: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = volunteers[volunteers["smooth-or-featured_total-votes"] >= MIN_VOTES]
    strat = df[FIRST_QUESTION].idxmax(axis=1)
    train_idx, rest_idx = train_test_split(df.index, test_size=0.20, stratify=strat, random_state=SEED)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.50, stratify=strat.loc[rest_idx], random_state=SEED)
    return {"train": volunteers.loc[train_idx], "val": volunteers.loc[val_idx], "test": volunteers.loc[test_idx]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path,
                   default=Path(os.environ.get("GZD5_DATA_DIR", Path.home() / "project_data" / "aa_gz_decals")))
    p.add_argument("--out-dir", type=Path, default=None, help="default: --data-dir")
    p.add_argument("--check", action="store_true",
                   help="compare with the split parquets already in --data-dir instead of writing")
    p.add_argument("--overwrite", action="store_true", help="replace split parquets that already exist")
    args = p.parse_args(argv)
    out_dir = args.out_dir or args.data_dir
    existing = [f for f in out_dir.glob("decals_split_*.parquet")] if out_dir.exists() else []
    if existing and not (args.check or args.overwrite):
        p.error(f"split parquets already exist in {out_dir}; use --check to compare with them "
                "(or --overwrite to replace them)")

    splits = make_splits(pd.read_parquet(args.data_dir / "gz_decals_volunteers_5.parquet"))
    for name, split in splits.items():
        path = out_dir / f"decals_split_{name}.parquet"
        if args.check:
            same = pd.read_parquet(path, columns=["iauname"])["iauname"].tolist() == split["iauname"].tolist()
            print(f"{name:5s} {len(split):7d} galaxies  {'identical to' if same else 'DIFFERENT from'} {path}")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            split.to_parquet(path, index=False)
            print(f"{name:5s} {len(split):7d} galaxies -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
