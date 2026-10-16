"""The three views the AugMix Jensen-Shannon loss needs: (clean, AugMix, AugMix) of the same image.

Goes last in the training pipeline, after the crop modality, so the clean view is the cropped (and, for
`wt`, flipped and rotated) image and both AugMix views share its geometry, as in Hendrycks et al. (2020).
Pair with the `gzd5_augmix_task` task, whose `dense_views` collate stacks the views.
"""
from __future__ import annotations

from dataclasses import dataclass

from optastra.transforms import Compose, Transform
from optastra.transforms.multiview import MultiViewTransform

__all__ = ["AugMixViewsConfig"]


@dataclass
class AugMixViewsConfig:
    augmix: str = "augmix"      # registered AugMix preset used for both augmented views


def augmix_views(cfg: AugMixViewsConfig) -> MultiViewTransform:
    augmix = Transform.create(cfg.augmix)
    return MultiViewTransform([Compose([]), augmix, augmix])


for _suffix in ("", "_weak", "_strong"):
    Transform.register(augmix_views, config=AugMixViewsConfig(augmix=f"augmix{_suffix}"),
                       name=f"augmix_views{_suffix}")
