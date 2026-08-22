from __future__ import annotations

from torch import Tensor, nn


class SharedRSIATAdapter(nn.Module):
    """A single adapter reused for every task; parameter count must not grow."""

    def __init__(self, feature_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, feature_dim)

    def forward(self, features: Tensor) -> Tensor:
        return features + self.up(self.activation(self.down(features)))
