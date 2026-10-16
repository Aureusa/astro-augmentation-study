"""GZD5Task with the Jensen-Shannon consistency loss of AugMix (Hendrycks et al. 2020, arXiv:1912.02781).

Every training image arrives as three views -- the clean image (after the crop modality) and two independent
AugMix views of it (transforms/augmix_views.py), stacked view-major by the `dense_views` collate:

    inputs = [clean_1..clean_B, aug1_1..aug1_B, aug2_1..aug2_B]          (3B, C, H, W)

The model sees all 3B images in one forward pass, as in the reference implementation. The loss is

    L = sum_q DM-NLL_q(clean)  +  js_weight * sum_q mean_i JS(p_clean, p_aug1, p_aug2)

where p = alpha / alpha_0 is the predicted vote-fraction distribution of question q and
JS(p_1..p_V) = 1/V sum_v KL(p_v || M), M = 1/V sum_v p_v. The JS term needs no labels, so it is applied to
every galaxy and every question, including questions nobody answered. Validation and test batches have a
single view, so their loss is exactly GZD5Task's and stays comparable with every other run.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch

from optastra.nn.features import HeadOutput
from optastra.tasks import Task

import data.collate  # noqa: F401  (registers the `dense_views` collate)

from .gz_decals import GZD5Task, GZD5TaskConfig


@dataclass
class GZD5AugMixTaskConfig(GZD5TaskConfig):
    js_weight: float = 12.0     # lambda of Hendrycks et al. (2020)


class GZD5Task_AugMix(GZD5Task):

    collate = "dense_views"

    def __init__(self, cfg: GZD5AugMixTaskConfig = GZD5AugMixTaskConfig()):
        super().__init__(cfg)

    def compute_losses(self, raw_preds: HeadOutput, targets: Mapping[str, Any]):
        batch_size = next(iter(targets.values())).shape[0]
        n_views = raw_preds.logits.shape[0] // batch_size
        if n_views == 1:                                   # val / test: plain GZD5 loss
            return super().compute_losses(raw_preds, targets)

        views = raw_preds.logits.view(n_views, batch_size, -1)
        losses = super().compute_losses(replace(raw_preds, logits=views[0]), targets)

        # In fp32: the KL terms of nearly identical distributions are tiny differences of logs.
        alphas = [self.logits_to_alphas(v.float()) for v in views]
        js = 0.0
        for question in self.questions:
            p = torch.stack([a[question.name] / a[question.name].sum(dim=-1, keepdim=True) for a in alphas])
            log_m = p.mean(dim=0).clamp(1e-7, 1.0).log()
            js = js + (p * (p.log() - log_m)).sum(dim=-1).mean()   # mean over views and galaxies
        losses["js"] = self.cfg.js_weight * js
        return losses


@Task.register(config=GZD5AugMixTaskConfig())
def gzd5_augmix_task(cfg: GZD5AugMixTaskConfig) -> GZD5Task_AugMix: return GZD5Task_AugMix(cfg)
