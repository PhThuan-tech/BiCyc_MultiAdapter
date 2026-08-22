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

    @property
    def bidirectional(self) -> Tensor:
        return self.backward + self.forward

    @property
    def cycle(self) -> Tensor:
        return self.cycle_new + self.cycle_old

    def total(self, lambda_bi: float, lambda_cyc: float) -> Tensor:
        return lambda_bi * self.bidirectional + lambda_cyc * self.cycle


def bicyc_loss(module: BidirectionalCycle, old_features: Tensor, new_features: Tensor) -> BiCycLoss:
    """BiCyc Eq. (6)--(8), with the paper's intended gradient gates.

    The backward term is the only BiCyc term allowed to update the new encoder.
    Cycle inputs are detached to make the cycle a projector-only stabilizer.
    """
    old = old_features.detach()
    new_target = new_features.detach()
    return BiCycLoss(
        backward=F.mse_loss(module.new_to_old(new_features), old),
        forward=F.mse_loss(module.old_to_new(old), new_target),
        cycle_new=F.mse_loss(module.old_to_new(module.new_to_old(new_target)), new_target),
        cycle_old=F.mse_loss(module.new_to_old(module.old_to_new(old)), old),
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
