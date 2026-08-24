"""Feature-distribution controls proposed for the KeepLoRA + BiCyc hybrid.

This is the proposed adaptive gate, not a loss claimed by either source paper.
It stores no old raw samples.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class DiagonalGaussian:
    mean: Tensor
    variance: Tensor


def estimate_diagonal_gaussian(features: Tensor, epsilon: float = 1e-5) -> DiagonalGaussian:
    if features.ndim != 2:
        raise ValueError("features must have shape [batch, dimension].")
    return DiagonalGaussian(features.mean(0), features.var(0, unbiased=False).clamp_min(epsilon))


def symmetric_gaussian_kl(left: DiagonalGaussian, right: DiagonalGaussian) -> Tensor:
    """Symmetric KL of diagonal Gaussians: O(d), stable for ViT features."""
    left_right = 0.5 * (
        left.variance / right.variance
        + (right.mean - left.mean).square() / right.variance
        - 1
        + right.variance.log()
        - left.variance.log()
    ).sum()
    right_left = 0.5 * (
        right.variance / left.variance
        + (left.mean - right.mean).square() / left.variance
        - 1
        + left.variance.log()
        - right.variance.log()
    ).sum()
    return 0.5 * (left_right + right_left)


def adaptive_alignment_weight(
    old_features: Tensor,
    new_features: Tensor,
    lambda_min: float,
    lambda_max: float,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return detached ``(lambda_adaptive, distance_per_dim, distance_raw)``.

    The gate grows with feature drift so that stronger distribution shift raises the
    stabilizer weight instead of collapsing towards ``lambda_min``. This is the
    practical fix for the smoke-test failure mode where the model keeps plasticity
    but forgets old tasks too quickly.

    The raw symmetric KL is a *sum* over ``feature_dim`` terms, so its magnitude
    scales with the model width: ViT-B (768 dims) produces values in the hundreds,
    which makes ``exp(-distance / temperature) ~ 0`` and pins the gate at
    ``lambda_max`` for every task -- i.e. the "adaptive" proposal degenerates into a
    fixed maximum stabilizer. The gate is therefore driven by the *per-dimension*
    mean KL, which is comparable across models of different feature widths, while
    the raw sum is still returned for logging.
    """
    if lambda_min < 0 or lambda_max < lambda_min or temperature <= 0:
        raise ValueError("Require 0 <= lambda_min <= lambda_max and temperature > 0.")
    old = estimate_diagonal_gaussian(old_features.detach())
    new = estimate_diagonal_gaussian(new_features.detach())
    distance = symmetric_gaussian_kl(old, new).clamp_min(0.0)
    feature_dim = max(int(old_features.shape[1]), 1)
    distance_per_dim = distance / feature_dim
    # Increase the stabilizer exactly when the old/new feature distributions drift apart;
    # this keeps the gate from saturating at the minimum when stability is most needed.
    weight = lambda_min + (lambda_max - lambda_min) * (1.0 - (-distance_per_dim / temperature).exp())
    return weight.detach(), distance_per_dim.detach(), distance.detach()
