"""Collate functions for astro_augment tasks (registered in optastra's CollateFn registry)."""
from __future__ import annotations

import torch
from torch.utils.data._utils.collate import default_collate as torch_default_collate

from optastra.data.collate import CollateFn, dense
from optastra.data.sample import Sample

__all__ = ["dense_views"]


@CollateFn.register
def dense_views(samples: list[Sample]) -> dict:
    """`dense`, except that samples carrying `views` (MultiViewTransform) are stacked view-major into one
    (V * B, C, H, W) tensor: view 0 of every sample, then view 1, ... Targets stay (B, ...), one per image.
    Samples without views (val / test) are collated exactly like `dense`."""
    if samples[0].views is None:
        return dense(samples)
    n_views = len(samples[0].views)
    inputs = torch.cat([torch.stack([s.views[v] for s in samples]) for v in range(n_views)])
    targets = {k: torch_default_collate([s.target[k] for s in samples]) for k in samples[0].target}
    return {"inputs": inputs, "targets": targets}
