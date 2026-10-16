from __future__ import annotations

from dataclasses import dataclass

import torch

from optastra.transforms import BatchTransform


__all__ = ["MixUp", "CutMix"]


@dataclass
class MixUpConfig:
    alpha: float = 0.2
    p: float = 0.5


class MixUp(BatchTransform):
    def __init__(self, cfg: MixUpConfig = MixUpConfig()):
        self.cfg = cfg

    def __call__(self, batch):
        if torch.rand(()) >= self.cfg.p:
            return batch

        images = batch["inputs"]
        targets = batch["targets"]

        lam = torch.distributions.Beta(
            self.cfg.alpha,
            self.cfg.alpha,
        ).sample().to(images.device)

        perm = torch.randperm(
            images.shape[0],
            device=images.device,
        )

        # Mix images.
        batch["inputs"] = (
            lam * images
            + (1.0 - lam) * images[perm]
        )

        # Mix every GZD5 question's vote counts.
        batch["targets"] = {
            question_name: (
                lam * counts
                + (1.0 - lam) * counts[perm]
            )
            for question_name, counts in targets.items()
        }

        return batch


@BatchTransform.register(config=MixUpConfig())
def mixup_decals(cfg):
    return MixUp(cfg)

@BatchTransform.register(config=MixUpConfig())
def mixup_decals_weak(cfg):
    cfg.alpha = 0.05
    cfg.p = 0.25
    return MixUp(cfg)
 
 
@BatchTransform.register(config=MixUpConfig())
def mixup_decals_strong(cfg):
    cfg.alpha = 1.0
    cfg.p = 0.8
    return MixUp(cfg)


@dataclass
class CutMixConfig:
    alpha: float = 1.0
    p: float = 0.5


def _rand_bbox(
    h: int,
    w: int,
    lam: float,
) -> tuple[int, int, int, int]:
    cut_ratio = (1.0 - lam) ** 0.5

    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cy = torch.randint(h, (1,)).item()
    cx = torch.randint(w, (1,)).item()

    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, h)

    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, w)

    return y1, y2, x1, x2


class CutMix(BatchTransform):
    def __init__(self, cfg: CutMixConfig = CutMixConfig()):
        self.cfg = cfg

    def __call__(self, batch):
        if torch.rand(()) >= self.cfg.p:
            return batch

        images = batch["inputs"]
        targets = batch["targets"]

        B, _, H, W = images.shape

        perm = torch.randperm(
            B,
            device=images.device,
        )

        lam = torch.distributions.Beta(
            self.cfg.alpha,
            self.cfg.alpha,
        ).sample().item()

        y1, y2, x1, x2 = _rand_bbox(
            H,
            W,
            lam,
        )

        # Replace patch.
        images[:, :, y1:y2, x1:x2] = (
            images[perm, :, y1:y2, x1:x2]
        )

        # Recalculate lambda from actual patch.
        patch_area = (y2 - y1) * (x2 - x1)

        lam_actual = 1.0 - (
            patch_area / float(H * W)
        )

        batch["inputs"] = images

        # Mix vote counts using the same spatial fraction.
        batch["targets"] = {
            question_name: (
                lam_actual * counts
                + (1.0 - lam_actual) * counts[perm]
            )
            for question_name, counts in targets.items()
        }

        return batch


@BatchTransform.register(config=CutMixConfig())
def cutmix_decals(cfg):
    return CutMix(cfg)


@BatchTransform.register(config=CutMixConfig())
def cutmix_decals_weak(cfg):
    cfg.p = 0.25
    cfg.alpha = 0.5
    return CutMix(cfg)
 
 
@BatchTransform.register(config=CutMixConfig())
def cutmix_decals_strong(cfg):
    cfg.p = 0.8
    cfg.alpha = 2.0  # concentrates lam near 0.5 -> a consistently sized patch, not occasional huge swaps
    return CutMix(cfg)
