"""Loss utilities shared by the two research directions."""

from __future__ import annotations

from torch import Tensor


def warmup_weight(target: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return target
    return target * min(1.0, epoch / warmup_epochs)


def representation_steering_loss(features: Tensor, labels: Tensor) -> Tensor:
    """Reserved for RSIAT (direction 2)."""
    raise NotImplementedError("Direction 2 is intentionally deferred.")
