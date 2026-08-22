import torch

from bicyc_multiadapter.models.alignment.bicyc import BidirectionalCycle, bicyc_loss
from bicyc_multiadapter.models.alignment.distribution import adaptive_alignment_weight


def test_cycle_loss_does_not_backpropagate_to_new_features() -> None:
    module = BidirectionalCycle(4)
    old = torch.randn(3, 4)
    new = torch.randn(3, 4, requires_grad=True)
    terms = bicyc_loss(module, old, new)
    terms.cycle.backward()
    assert new.grad is None


def test_more_similar_distributions_receive_larger_adaptive_weight() -> None:
    old = torch.randn(16, 4)
    near, _ = adaptive_alignment_weight(old, old + 0.01, 0.1, 1.0, 1.0)
    far, _ = adaptive_alignment_weight(old, old + 10, 0.1, 1.0, 1.0)
    assert near > far
