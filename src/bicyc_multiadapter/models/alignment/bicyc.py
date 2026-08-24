"""BiCyc alignment from Xu & Krawczyk, ICLR 2026."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _Projector(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int | None) -> None:
        super().__init__()
        self.net = (
            nn.Linear(dimension, dimension)
            if hidden_dimension is None
            else nn.Sequential(nn.Linear(dimension, hidden_dimension), nn.GELU(), nn.Linear(hidden_dimension, dimension))
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class BidirectionalCycle(nn.Module):
    """A: old -> new and D: new -> old, learned during the current task."""

    def __init__(self, feature_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.old_to_new = _Projector(feature_dim, hidden_dim)
        self.new_to_old = _Projector(feature_dim, hidden_dim)

    def forward(self, old_features: Tensor, new_features: Tensor) -> dict[str, Tensor]:
        return {
            "pred_new": self.old_to_new(old_features),
            "pred_old": self.new_to_old(new_features),
            "cycle_new": self.old_to_new(self.new_to_old(new_features)),
            "cycle_old": self.new_to_old(self.old_to_new(old_features)),
        }


@dataclass
class BiCycLoss:
    backward: Tensor
    forward: Tensor
    cycle_new: Tensor
    cycle_old: Tensor
    isometric: Tensor | None = None

    @property
    def bidirectional(self) -> Tensor:
        return self.backward + self.forward

    @property
    def cycle(self) -> Tensor:
        return self.cycle_new + self.cycle_old

    def total(self, lambda_bi: float, lambda_cyc: float, lambda_iso: float = 0.0) -> Tensor:
        loss = lambda_bi * self.bidirectional + lambda_cyc * self.cycle
        if lambda_iso > 0 and self.isometric is not None:
            loss = loss + lambda_iso * self.isometric
        return loss


def isometric_regularization_loss(source_features: Tensor, mapped_features: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Penalize norm distortion and direction collapse of affine transport map A."""
    src_norm = torch.linalg.norm(source_features, dim=1).clamp_min(epsilon)
    mapped_norm = torch.linalg.norm(mapped_features, dim=1).clamp_min(epsilon)
    norm_loss = (mapped_norm / src_norm - 1.0).square().mean()
    cosine = (source_features * mapped_features).sum(dim=1) / (src_norm * mapped_norm)
    direction_loss = (1.0 - cosine).clamp_min(0.0).mean()
    return norm_loss + direction_loss


def bicyc_loss(module: BidirectionalCycle, old_features: Tensor, new_features: Tensor) -> BiCycLoss:
    """BiCyc Eq. (6)--(8), with the paper's intended gradient gates.

    The backward term is the only BiCyc term allowed to update the new encoder.
    Cycle inputs are detached to make the cycle a projector-only stabilizer.
    """
    old = old_features.detach()
    new_target = new_features.detach()
    pred_new = module.old_to_new(old)
    pred_old = module.new_to_old(new_features)
    iso_loss = isometric_regularization_loss(old, pred_new)
    return BiCycLoss(
        backward=F.mse_loss(pred_old, old),
        forward=F.mse_loss(pred_new, new_target),
        cycle_new=F.mse_loss(module.old_to_new(module.new_to_old(new_target)), new_target),
        cycle_old=F.mse_loss(module.new_to_old(module.old_to_new(old)), old),
        isometric=iso_loss,
    )


def robust_anti_collapse_loss(features: Tensor, beta: float = 0.1, shrinkage: float = 1e-3) -> Tensor:
    """Numerically stable covariance anti-collapse regularizer from BiCyc."""
    centered = features - features.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(features.shape[0] - 1, 1)
    covariance = 0.5 * (covariance + covariance.T)
    dimension = covariance.shape[0]
    covariance = covariance + (shrinkage * covariance.diagonal().mean() + 1e-6) * torch.eye(
        dimension, device=features.device, dtype=features.dtype
    )
    diagonal = torch.linalg.cholesky(covariance).diagonal()
    return -torch.minimum(diagonal, torch.full_like(diagonal, beta)).mean()
