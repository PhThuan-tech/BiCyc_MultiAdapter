from __future__ import annotations

from torch import Tensor, nn


class BidirectionalResidualAutoencoder(nn.Module):
    """Training-only Bi-RAE; it must be bypassed at inference."""

    def __init__(self, feature_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(feature_dim, latent_dim)
        self.to_current = nn.Linear(latent_dim, feature_dim)
        self.to_previous = nn.Linear(feature_dim, feature_dim)

    def forward(self, previous_features: Tensor, current_features: Tensor) -> dict[str, Tensor]:
        projected_current = self.to_current(self.encoder(previous_features))
        reconstructed_previous = self.to_previous(current_features)
        return {"projected_current": projected_current, "reconstructed_previous": reconstructed_previous}
