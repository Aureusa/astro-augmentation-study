"""Experiment registry: every experiment name -> its train / val transforms and batch transforms.

Names follow the run directories in runs/, so a new run is directly comparable with the old one of the same
name. Every pipeline starts with `to_float`.

    <policy>[_weak|_strong]_<modality>

    modality  rc  random 224 crop                         (fixed size, random position)
              wt  Walmsley transforms: hflip, vflip, rotation (0-180, reflect), random 224 crop
    policy    applied after the modality's crop; no strength suffix = the standard (paper) preset

    augmix_jsd[_weak|_strong]_<modality> is AugMix trained with its Jensen-Shannon consistency loss
    (task gzd5_augmix_task, task/gz_decals_augmix.py).

RandomResizedCrop (`rrc`) experiments are deliberately not registered: they trained on zoomed-in views but
validated on plain center crops (a train/test scale mismatch), so they are not a fair comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Experiment", "EXPERIMENTS", "SWEEP_EXPERIMENTS", "AUGMIX_JSD_SWEEP_EXPERIMENTS",
           "AUGMIX_JSD_STRENGTH_EXPERIMENTS", "get_experiment"]


@dataclass(frozen=True)
class Experiment:
    name: str
    train: tuple[str, ...]                     # per-sample Transform names (after to_float)
    val: tuple[str, ...] = ("crop",)           # per-sample Transform names for val / test (after to_float)
    batch: tuple[str, ...] = ()                # BatchTransform names, applied to every training batch
    in_channels: int = 3
    task: str = "gzd5_task"                    # registered Task name
    notes: str = ""

    def describe(self) -> dict:
        return {"name": self.name, "train": ["to_float", *self.train], "val": ["to_float", *self.val],
                "batch": list(self.batch), "in_channels": self.in_channels, "task": self.task, "notes": self.notes}


CROP = {"rc": ("random_crop",), "wt": ("walmsley_transforms",)}

# family -> {strength: (per-sample Transform name or None, BatchTransform name or None)}
POLICIES = {
    "rand_augment_all_ops": {"strong": ("rand_augment_all_ops", None)},          # N=2, M=9: the standard preset
    "randaugment": {"weak": ("randaugment_weak", None), "moderate": ("randaugment_moderate", None)},
    "auto_augment": {s: (f"auto_augment{suf}", None) for s, suf in [("weak", "_weak"), ("moderate", ""), ("strong", "_strong")]},
    "trivial_augment": {s: (f"trivial_augment{suf}", None) for s, suf in [("weak", "_weak"), ("moderate", ""), ("strong", "_strong")]},
    "augmix": {s: (f"augmix{suf}", None) for s, suf in [("weak", "_weak"), ("moderate", ""), ("strong", "_strong")]},
    "cutmix": {s: (None, f"cutmix_decals{suf}") for s, suf in [("weak", "_weak"), ("moderate", ""), ("strong", "_strong")]},
    "mixup": {s: (None, f"mixup_decals{suf}") for s, suf in [("weak", "_weak"), ("moderate", ""), ("strong", "_strong")]},
}
# the "standard" strength of each family = the preset whose run name has no strength suffix
STANDARD = {"rand_augment_all_ops": "strong", "auto_augment": "moderate", "trivial_augment": "moderate",
            "augmix": "moderate", "cutmix": "moderate", "mixup": "moderate"}


def _build() -> dict[str, Experiment]:
    exps = [
        Experiment("baseline_A", train=("resize",), val=("resize",), notes="resize 424 -> 224, no augmentation"),
        Experiment("baseline_B", train=("crop",), notes="center crop 224, no augmentation"),
        Experiment("baseline_rc", train=CROP["rc"], notes="random 224 crop only"),
        Experiment("walmsley_transforms", train=CROP["wt"], notes="flips + rotation + random crop"),
        Experiment("walmsley_transforms_gray", train=("to_grayscale", *CROP["wt"]), val=("to_grayscale", "crop"),
                   in_channels=1, notes="channel-mean grayscale, then Walmsley transforms"),
    ]
    for family, presets in POLICIES.items():
        for strength, (per_sample, batch) in presets.items():
            if family in STANDARD and strength == STANDARD[family]:
                stem = family
            elif family == "randaugment":
                stem = f"randaugment_{strength}"                    # historical names: randaugment_weak_rc, ...
            else:
                stem = f"{family}_{strength}"
            for modality, crop in CROP.items():
                exps.append(Experiment(
                    f"{stem}_{modality}",
                    train=crop + ((per_sample,) if per_sample else ()),
                    batch=(batch,) if batch else (),
                    notes=f"{family} ({strength}) after {modality}",
                ))
    # AugMix with the Jensen-Shannon consistency loss: same presets and names as augmix_*, family augmix_jsd
    for strength, suffix in [("weak", "_weak"), ("moderate", ""), ("strong", "_strong")]:
        stem = "augmix_jsd" if strength == "moderate" else f"augmix_jsd_{strength}"
        for modality, crop in CROP.items():
            exps.append(Experiment(
                f"{stem}_{modality}", train=crop + (f"augmix_views{suffix}",), task="gzd5_augmix_task",
                notes=f"augmix ({strength}) + Jensen-Shannon consistency loss after {modality}",
            ))
    return {e.name: e for e in exps}


EXPERIMENTS = _build()

# The training-set-size sweep: the no-policy references and every family at its standard strength, in both
# crop modalities. Ordered by priority -- the sweep runs them in this order within each fraction.
SWEEP_EXPERIMENTS = [
    "walmsley_transforms", "baseline_B", "baseline_rc",
    *[f"{family}_wt" for family in ["augmix", "trivial_augment", "mixup", "auto_augment", "rand_augment_all_ops", "cutmix"]],
    *[f"{family}_rc" for family in ["augmix", "trivial_augment", "mixup", "auto_augment", "rand_augment_all_ops", "cutmix"]],
]

# AugMix + Jensen-Shannon loss (train_augmix_jsd.slurm): the standard preset over every training-set size, and
# the weak / strong presets on the full training set only, as in the strength study.
AUGMIX_JSD_SWEEP_EXPERIMENTS = ["augmix_jsd_wt", "augmix_jsd_rc"]
AUGMIX_JSD_STRENGTH_EXPERIMENTS = ["augmix_jsd_weak_wt", "augmix_jsd_weak_rc",
                                   "augmix_jsd_strong_wt", "augmix_jsd_strong_rc"]


def get_experiment(name: str) -> Experiment:
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}; known: {', '.join(sorted(EXPERIMENTS))}")
    return EXPERIMENTS[name]
