"""GZD5Task_AugMix: the Jensen-Shannon term and its interplay with the plain GZD5 loss."""
import pytest
import torch

from optastra.data.sample import Sample
from optastra.nn.features import HeadOutput

from data.collate import dense_views
from task.gz_decals import GZD5Task
from task.gz_decals_augmix import GZD5Task_AugMix

B = 8


@pytest.fixture
def batch():
    torch.manual_seed(0)
    task = GZD5Task_AugMix()
    targets = {q.name: torch.randint(0, 5, (B, len(q.answers))).float() for q in task.questions}
    return task, targets, torch.randn(B, task.n_outputs)


def test_single_view_is_the_plain_loss(batch):
    task, targets, logits = batch
    plain = GZD5Task().compute_losses(HeadOutput(logits=logits), targets)
    losses = task.compute_losses(HeadOutput(logits=logits), targets)
    assert losses.keys() == plain.keys()
    for k in plain:
        assert torch.allclose(losses[k], plain[k])


def test_identical_views_have_no_js(batch):
    task, targets, logits = batch
    losses = task.compute_losses(HeadOutput(logits=logits.repeat(3, 1)), targets)
    assert losses["js"].abs() < 1e-5
    plain = GZD5Task().compute_losses(HeadOutput(logits=logits), targets)
    for k in plain:
        assert torch.allclose(losses[k], plain[k])


def test_different_views_have_positive_js(batch):
    task, targets, logits = batch
    views = torch.cat([logits, torch.randn(2 * B, task.n_outputs)])
    assert task.compute_losses(HeadOutput(logits=views), targets)["js"] > 0


def test_dense_views_stacks_view_major():
    samples = [Sample(views=[torch.full((1, 2, 2), 10.0 * v + i) for v in range(3)], target={"q": torch.tensor([i])})
               for i in range(4)]
    batch = dense_views(samples)
    assert batch["inputs"].shape == (12, 1, 2, 2)
    assert batch["inputs"][:, 0, 0, 0].tolist() == [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]
    assert batch["targets"]["q"].shape == (4, 1)
