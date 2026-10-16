"""
Sanity checks for GZD5Task's Dirichlet-multinomial likelihood and the
logits -> alpha mapping. These catch the two easiest ways to silently get
this loss wrong: sign errors in the log-likelihood, and alpha ranges that
don't match [alpha_min, alpha_max].
"""
import math

import pytest
import torch

from task.gz_decals import GZD5Task, GZD5TaskConfig
from task.gdz_5_schema import GZD5_QUESTIONS


@pytest.fixture
def task():
    return GZD5Task(GZD5TaskConfig(alpha_min=1.0, alpha_max=100.0, ignore_unanswered=True))


# ---------------------------------------------------------------------------
# dirichlet_multinomial_nll
# ---------------------------------------------------------------------------

class TestDirichletMultinomialNLL:

    def test_zero_votes_gives_zero_nll(self, task):
        """n=0 -> no observations -> likelihood of the empty observation is 1,
        so NLL must be exactly 0 regardless of alpha."""
        alpha = torch.tensor([[2.0, 5.0, 1.0]])
        counts = torch.tensor([[0.0, 0.0, 0.0]])
        nll = task.dirichlet_multinomial_nll(alpha, counts)
        assert torch.allclose(nll, torch.zeros_like(nll), atol=1e-5)

    def test_nll_is_nonnegative(self, task):
        """NLL of a valid probability must never be negative."""
        torch.manual_seed(0)
        alpha = torch.rand(16, 3) * 99 + 1  # in [1, 100], matching alpha_min/max
        counts = torch.randint(0, 20, (16, 3)).float()
        nll = task.dirichlet_multinomial_nll(alpha, counts)
        assert (nll >= -1e-5).all()

    def test_matches_closed_form_binomial_case(self, task):
        """For a 2-answer question with alpha = [a, a] (symmetric), the
        Dirichlet-multinomial reduces to the Beta-binomial distribution.
        Check against scipy's closed form (or a manual log-Beta-binomial)
        for a small n where the two must agree exactly."""
        a = 3.0
        alpha = torch.tensor([[a, a]])
        k, n = 2, 5  # 2 successes out of 5 votes
        counts = torch.tensor([[float(k), float(n - k)]])

        nll = task.dirichlet_multinomial_nll(alpha, counts).item()

        # Beta-binomial pmf: C(n,k) * B(k+a, n-k+a) / B(a,a)
        log_binom_coef = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        log_beta_num = math.lgamma(k + a) + math.lgamma(n - k + a) - math.lgamma(n + 2 * a)
        log_beta_den = math.lgamma(a) + math.lgamma(a) - math.lgamma(2 * a)
        expected_log_prob = log_binom_coef + log_beta_num - log_beta_den
        expected_nll = -expected_log_prob

        assert nll == pytest.approx(expected_nll, abs=1e-4)

    def test_concentrated_alpha_favors_matching_counts(self, task):
        """A sharply concentrated alpha (all mass on answer 0) should give
        much lower NLL to counts that are also concentrated on answer 0
        than to counts concentrated on answer 1 -- a basic direction check."""
        alpha = torch.tensor([[95.0, 1.0, 1.0]])  # ~all mass on answer 0
        counts_matching = torch.tensor([[10.0, 0.0, 0.0]])
        counts_mismatched = torch.tensor([[0.0, 10.0, 0.0]])

        nll_matching = task.dirichlet_multinomial_nll(alpha, counts_matching)
        nll_mismatched = task.dirichlet_multinomial_nll(alpha, counts_mismatched)

        assert nll_matching.item() < nll_mismatched.item()

    def test_batch_shape_preserved(self, task):
        alpha = torch.rand(8, 4) * 50 + 1
        counts = torch.randint(0, 10, (8, 4)).float()
        nll = task.dirichlet_multinomial_nll(alpha, counts)
        assert nll.shape == (8,)


# ---------------------------------------------------------------------------
# logits_to_alphas
# ---------------------------------------------------------------------------

class TestLogitsToAlphas:

    def test_output_keys_match_questions(self, task):
        B = 4
        logits = torch.randn(B, task.n_outputs)
        alphas = task.logits_to_alphas(logits)
        assert set(alphas.keys()) == {q.name for q in GZD5_QUESTIONS}

    def test_alpha_shapes_match_answer_counts(self, task):
        B = 4
        logits = torch.randn(B, task.n_outputs)
        alphas = task.logits_to_alphas(logits)
        for q in GZD5_QUESTIONS:
            assert alphas[q.name].shape == (B, len(q.answers))

    def test_alpha_within_configured_range(self, task):
        """Every alpha value must lie in [alpha_min, alpha_max] by
        construction (softmax output in [0,1], linearly rescaled)."""
        B = 32
        logits = torch.randn(B, task.n_outputs) * 5  # wide range, including extreme logits
        alphas = task.logits_to_alphas(logits)
        for q in GZD5_QUESTIONS:
            a = alphas[q.name]
            assert (a >= task.cfg.alpha_min - 1e-4).all()
            assert (a <= task.cfg.alpha_max + 1e-4).all()

    def test_uniform_logits_give_uniform_alpha(self, task):
        """All-zero logits -> softmax is uniform -> every alpha within a
        question should be identical (the midpoint of [alpha_min, alpha_max]
        scaled by 1/num_answers, but critically: all equal)."""
        logits = torch.zeros(1, task.n_outputs)
        alphas = task.logits_to_alphas(logits)
        for q in GZD5_QUESTIONS:
            a = alphas[q.name][0]
            assert torch.allclose(a, a[0].expand_as(a), atol=1e-5)

    def test_wrong_logit_dim_raises(self, task):
        with pytest.raises(ValueError):
            task.logits_to_alphas(torch.randn(4, task.n_outputs - 1))

    def test_wrong_ndim_raises(self, task):
        with pytest.raises(ValueError):
            task.logits_to_alphas(torch.randn(task.n_outputs))  # missing batch dim


# ---------------------------------------------------------------------------
# compute_losses / reduce_losses integration
# ---------------------------------------------------------------------------

class TestComputeLosses:

    def test_all_questions_present_in_output(self, task):
        from optastra.nn.features import HeadOutput

        B = 4
        logits = torch.randn(B, task.n_outputs)
        raw_preds = HeadOutput(logits=logits)
        targets = {
            q.name: torch.randint(0, 10, (B, len(q.answers))).float()
            for q in GZD5_QUESTIONS
        }

        losses = task.compute_losses(raw_preds, targets)
        assert set(losses.keys()) == {f"{q.name}_dm" for q in GZD5_QUESTIONS}
        for v in losses.values():
            assert v.ndim == 0  # scalar per question, already reduced over batch

    def test_unanswered_questions_contribute_zero_when_ignored(self, task):
        """If every example has 0 votes for a question, that question's
        loss must be exactly 0 (not NaN, not skipped with a KeyError)."""
        from optastra.nn.features import HeadOutput

        B = 4
        logits = torch.randn(B, task.n_outputs)
        raw_preds = HeadOutput(logits=logits)
        targets = {q.name: torch.zeros(B, len(q.answers)) for q in GZD5_QUESTIONS}

        losses = task.compute_losses(raw_preds, targets)
        for v in losses.values():
            assert torch.isfinite(v)
            assert v.item() == pytest.approx(0.0, abs=1e-6)

    def test_reduce_losses_sums_all_questions(self, task):
        losses = {"a_dm": torch.tensor(1.5), "b_dm": torch.tensor(2.5)}
        total = task.reduce_losses(losses)
        assert total.item() == pytest.approx(4.0)

    def test_reduce_losses_empty_raises(self, task):
        with pytest.raises(ValueError):
            task.reduce_losses({})
