from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import torch

from optastra.tasks import Task, Stage
from optastra.nn.features import HeadOutput

from .gdz_5_schema import GZD5_QUESTIONS


@dataclass
class GZD5TaskConfig:
    alpha_min: float = 1.0
    alpha_max: float = 100.0

    # If True, questions with zero votes contribute zero loss.
    ignore_unanswered: bool = True


class GZD5Task(Task):

    required_fields = ("logits",)
    collate = "dense"

    def __init__(
        self,
        cfg: GZD5TaskConfig = GZD5TaskConfig(),
    ):
        self.cfg = cfg
        self.questions = GZD5_QUESTIONS

        self.n_outputs = sum(
            len(q.answers)
            for q in self.questions
        )

    # ---------------------------------------------------------
    # Batch validation
    # ---------------------------------------------------------

    def validate_batch(
        self,
        batch: Mapping[str, Any],
        stage: Stage = "train",
    ):
        if stage in ("train", "val", "test"):
            if "inputs" not in batch:
                raise ValueError(
                    "GZD5 batch must contain 'inputs'."
                )

            if "targets" not in batch:
                raise ValueError(
                    "GZD5 batch must contain 'targets'."
                )

            for question in self.questions:
                if question.name not in batch["targets"]:
                    raise ValueError(
                        f"Missing GZD5 target "
                        f"'{question.name}'."
                    )

        elif stage == "predict":

            if "inputs" not in batch:
                raise ValueError(
                    "GZD5 predict batch must contain 'inputs'."
                )

    # ---------------------------------------------------------
    # Inputs / targets
    # ---------------------------------------------------------

    def split_inputs_targets(
        self,
        batch: Mapping[str, Any],
        stage: Stage = "train",
    ):
        if stage in ("train", "val", "test"):
            return (
                batch["inputs"],
                batch["targets"],
            )

        return batch["inputs"], None

    # ---------------------------------------------------------
    # Target preprocessing
    # ---------------------------------------------------------

    def preprocess_targets(
        self,
        raw_targets: Mapping[str, Any],
    ):
        targets = {}

        for question in self.questions:

            counts = raw_targets[question.name]

            if counts.ndim != 2:
                raise ValueError(
                    f"{question.name}: expected "
                    f"[B, {len(question.answers)}], "
                    f"got {tuple(counts.shape)}"
                )

            if counts.shape[-1] != len(question.answers):
                raise ValueError(
                    f"{question.name}: expected "
                    f"{len(question.answers)} answers, "
                    f"got {counts.shape[-1]}"
                )

            if torch.any(counts < 0):
                raise ValueError(
                    f"{question.name}: vote counts "
                    f"cannot be negative."
                )

            targets[question.name] = counts.float()

        return targets

    # ---------------------------------------------------------
    # Model forward
    # ---------------------------------------------------------

    def forward_model(
        self,
        model,
        inputs: Mapping[str, Any],
    ) -> Any:

        return model(inputs)

    # ---------------------------------------------------------
    # Logits -> Dirichlet parameters
    # ---------------------------------------------------------

    def logits_to_alphas(
        self,
        logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        if logits.ndim != 2:
            raise ValueError(f"Expected logits [B, {self.n_outputs}], got {tuple(logits.shape)}")

        if logits.shape[-1] != self.n_outputs:
            raise ValueError(f"Expected {self.n_outputs} logits, got {logits.shape[-1]}")

        alphas = {}
        offset = 0

        for question in self.questions:
            n = len(question.answers)
            q_logits = logits[:, offset:offset + n]

            q_alpha = (
                self.cfg.alpha_min
                + (self.cfg.alpha_max - self.cfg.alpha_min) * torch.softmax(q_logits, dim=-1)
            )

            alphas[question.name] = q_alpha
            offset += n

        return alphas

    # ---------------------------------------------------------
    # Dirichlet-multinomial likelihood
    # ---------------------------------------------------------

    @staticmethod
    def dirichlet_multinomial_nll(
        alpha: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:

        counts = counts.to(dtype=alpha.dtype)

        n = counts.sum(dim=-1)
        alpha0 = alpha.sum(dim=-1)

        log_prob = (
            torch.lgamma(n + 1.0)
            - torch.lgamma(counts + 1.0).sum(dim=-1)

            + torch.lgamma(alpha0)
            - torch.lgamma(alpha0 + n)

            + (
                torch.lgamma(counts + alpha)
                - torch.lgamma(alpha)
            ).sum(dim=-1)
        )

        return -log_prob

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    def compute_losses(
        self,
        raw_preds: HeadOutput,
        targets: Mapping[str, Any],
    ):

        alphas = self.logits_to_alphas(
            raw_preds.logits
        )

        losses = {}

        for question in self.questions:

            name = question.name

            counts = targets[name]
            alpha = alphas[name]

            per_example_loss = (
                self.dirichlet_multinomial_nll(
                    alpha,
                    counts,
                )
            )

            if self.cfg.ignore_unanswered:

                n_votes = counts.sum(dim=-1)

                mask = n_votes > 0

                if mask.any():
                    loss = per_example_loss[mask].mean()
                else:
                    loss = per_example_loss.mean() * 0.0

            else:
                loss = per_example_loss.mean()

            losses[f"{name}_dm"] = loss

        return losses

    # ---------------------------------------------------------
    # Loss reduction
    # ---------------------------------------------------------

    def reduce_losses(
        self,
        losses: dict[str, torch.Tensor],
    ) -> torch.Tensor:

        if not losses:
            raise ValueError(
                "No GZD5 losses were produced."
            )

        # Original Zoobot conceptually sums
        # the question likelihoods.
        return sum(losses.values())

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    def compute_metrics(
        self,
        raw_preds: HeadOutput,
        targets: Mapping[str, Any],
    ):

        alphas = self.logits_to_alphas(
            raw_preds.logits
        )

        metrics = {}

        for question in self.questions:

            name = question.name

            alpha = alphas[name]
            counts = targets[name]

            vote_fraction = (
                counts
                / counts.sum(
                    dim=-1,
                    keepdim=True
                ).clamp_min(1.0)
            )

            predicted_fraction = (
                alpha / alpha.sum(
                    dim=-1,
                    keepdim=True
                )
            )

            n_votes = counts.sum(dim=-1)

            valid = n_votes > 0

            if valid.any():

                pred = predicted_fraction.argmax(
                    dim=-1
                )

                target = vote_fraction.argmax(
                    dim=-1
                )

                accuracy = (
                    pred[valid] == target[valid]
                ).float().mean()

                metrics[
                    f"{name}_accuracy"
                ] = accuracy

                rmse = torch.sqrt(
                    (
                        predicted_fraction[valid]
                        - vote_fraction[valid]
                    ).pow(2).mean()
                )

                metrics[
                    f"{name}_rmse"
                ] = rmse

            else:

                metrics[
                    f"{name}_accuracy"
                ] = 0.0

                metrics[
                    f"{name}_rmse"
                ] = 0.0

        return metrics

    # ---------------------------------------------------------
    # Prediction decoding
    # ---------------------------------------------------------

    def decode_predictions(
        self,
        raw_preds: HeadOutput,
    ):

        alphas = self.logits_to_alphas(
            raw_preds.logits
        )

        predictions = {}

        for question in self.questions:

            alpha = alphas[question.name]

            probability = (
                alpha
                / alpha.sum(
                    dim=-1,
                    keepdim=True
                )
            )

            predictions[question.name] = {
                "alpha": alpha,
                "probability": probability,
                "predicted_index": probability.argmax(
                    dim=-1
                ),
                "predicted_answer": [
                    question.answers[i]
                    for i in probability.argmax(
                        dim=-1
                    ).tolist()
                ],
            }

        return predictions


gzd5_task_config = {
    "gzd5_task": GZD5TaskConfig(
        alpha_min=1.0,
        alpha_max=100.0,
        ignore_unanswered=True,
    )
}


@Task.register(config=gzd5_task_config["gzd5_task"])
def gzd5_task(cfg: GZD5TaskConfig,) -> GZD5Task: return GZD5Task(cfg)
