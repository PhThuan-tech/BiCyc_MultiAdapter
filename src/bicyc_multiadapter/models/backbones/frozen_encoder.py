from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class FrozenFeatureEncoder(nn.Module, ABC):
    """Contract for a pretrained ViT/CLIP encoder whose parameters stay frozen."""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimension of the returned representation."""

    @abstractmethod
    def forward_features(self, images: Tensor) -> Tensor:
        """Return one feature vector per input image."""

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
