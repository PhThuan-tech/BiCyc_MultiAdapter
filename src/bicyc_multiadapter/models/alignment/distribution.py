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
    return channelwise_symmetric_gaussian_kl(left, right).sum()


def channelwise_symmetric_gaussian_kl(left: DiagonalGaussian, right: DiagonalGaussian) -> Tensor:
    """Per-dimension symmetric KL of diagonal Gaussians: shape [dimension]."""
    left_right = 0.5 * (
        left.variance / right.variance
        + (right.mean - left.mean).square() / right.variance
        - 1
        + right.variance.log()
        - left.variance.log()
    )
    right_left = 0.5 * (
        right.variance / left.variance
        + (left.mean - right.mean).square() / left.variance
        - 1
        + left.variance.log()
        - right.variance.log()
    )
    return (0.5 * (left_right + right_left)).clamp_min(0.0)


def adaptive_alignment_weight(
    old_features: Tensor,
    new_features: Tensor,
    lambda_min: float,
    lambda_max: float,
    temperature: float,
    channelwise: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return detached ``(lambda_adaptive, distance_per_dim, distance_raw)``.

    If ``channelwise=True``, ``lambda_adaptive`` has shape ``[feature_dim]``, giving
    fine-grained per-channel stabilizer weights. Otherwise, returns a scalar.
    """
    if lambda_min < 0 or lambda_max < lambda_min or temperature <= 0:
        raise ValueError("Require 0 <= lambda_min <= lambda_max and temperature > 0.")
    old = estimate_diagonal_gaussian(old_features.detach())
    new = estimate_diagonal_gaussian(new_features.detach())
    kl_per_channel = channelwise_symmetric_gaussian_kl(old, new)
    distance_raw = kl_per_channel.sum()
    feature_dim = max(int(old_features.shape[1]), 1)
    distance_per_dim = distance_raw / feature_dim
    if channelwise:
        weight = lambda_min + (lambda_max - lambda_min) * (1.0 - (-kl_per_channel / temperature).exp())
    else:
        weight = lambda_min + (lambda_max - lambda_min) * (1.0 - (-distance_per_dim / temperature).exp())
    return weight.detach(), distance_per_dim.detach(), distance_raw.detach()
