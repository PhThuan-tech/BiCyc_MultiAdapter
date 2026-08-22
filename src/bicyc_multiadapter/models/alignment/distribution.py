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
) -> tuple[Tensor, Tensor]:
    """Return detached (lambda_adaptive, distribution_distance)."""
    if lambda_min < 0 or lambda_max < lambda_min or temperature <= 0:
        raise ValueError("Require 0 <= lambda_min <= lambda_max and temperature > 0.")
    old = estimate_diagonal_gaussian(old_features.detach())
    new = estimate_diagonal_gaussian(new_features.detach())
    distance = symmetric_gaussian_kl(old, new)
    weight = lambda_min + (lambda_max - lambda_min) * (-distance / temperature).exp()
    return weight.detach(), distance.detach()
