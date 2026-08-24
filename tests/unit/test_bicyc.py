import pytest
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


def test_larger_drift_receives_larger_adaptive_weight() -> None:
    """The gate is an anti-collapse stabilizer: more drift => higher lambda_t.

    This is the intended direction (see distribution.py docstring); the historical
    test asserting the opposite direction was wrong and never matched the code.
    """
    old = torch.randn(16, 4)
    near, _, _ = adaptive_alignment_weight(old, old + 0.01, 0.1, 1.0, 1.0)
    far, _, _ = adaptive_alignment_weight(old, old + 10, 0.1, 1.0, 1.0)
    assert far > near
    assert 0.1 <= float(near) <= 1.0
    assert float(far) == pytest.approx(1.0)


def test_gate_is_driven_by_per_dimension_distance_not_the_raw_sum() -> None:
    """Regression: the raw symmetric KL is a sum over feature_dim, so for ViT-B
    (768 dims) it reached hundreds and saturated the gate at lambda_max for every
    task -- the "adaptive" proposal degenerated into a fixed maximum stabilizer."""
    torch.manual_seed(0)
    old = torch.randn(32, 768)
    shift = 0.2  # modest drift: per-dim KL ~ 0.02, raw sum ~ 15
    weight, distance_per_dim, distance_raw = adaptive_alignment_weight(old, old + shift, 0.4, 1.0, 1.0)
    # The gate must stay strictly below the ceiling so it can differentiate tasks.
    assert float(weight) < 1.0
    assert float(weight) > 0.4
    # distance_per_dim is exactly the raw sum normalised by the feature dimension.
    assert torch.allclose(distance_raw / 768, distance_per_dim)


def test_channelwise_adaptive_weight_returns_vector_gate() -> None:
    torch.manual_seed(0)
    old = torch.randn(32, 16)
    new = old.clone()
    new[:, :8] += 5.0  # large drift in first 8 channels, zero drift in last 8 channels
    weight, distance_per_dim, distance_raw = adaptive_alignment_weight(
        old, new, lambda_min=0.2, lambda_max=1.0, temperature=1.0, channelwise=True
    )
    assert weight.shape == (16,)
    # Drifted channels receive high stabilizer weight; unchanged channels stay near lambda_min
    assert weight[:8].mean() > weight[8:].mean()
    assert (weight >= 0.2).all() and (weight <= 1.0).all()


def test_isometric_loss_in_bicyc() -> None:
    module = BidirectionalCycle(8)
    old = torch.randn(10, 8)
    new = torch.randn(10, 8)
    terms = bicyc_loss(module, old, new)
    assert terms.isometric is not None
    assert terms.isometric.item() >= 0.0
    total = terms.total(lambda_bi=1.0, lambda_cyc=1.0, lambda_iso=0.5)
    assert total.item() > 0.0
