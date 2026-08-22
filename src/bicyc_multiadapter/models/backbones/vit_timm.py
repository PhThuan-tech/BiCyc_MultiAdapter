"""Frozen timm Vision-Transformer encoder used by Direction 1."""

from __future__ import annotations

import timm
from torch import Tensor, nn

from .frozen_encoder import FrozenFeatureEncoder


class TimmViTEncoder(FrozenFeatureEncoder):
    """Pretrained ViT with its classification head removed; parameters never unfreeze."""

    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = True) -> None:
        super().__init__()
        # num_classes=0 makes timm return the pooled representation instead of logits.
        self.network = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.freeze()

    @property
    def feature_dim(self) -> int:
        return int(self.network.num_features)

    def forward_features(self, images: Tensor) -> Tensor:
        return self.network(images)

    def candidate_linears(self) -> list[tuple[str, nn.Linear]]:
        """Block-internal projections available for LoRA patching, e.g. ``blocks.5.attn.qkv``."""
        return [(name, module) for name, module in self.network.named_modules() if isinstance(module, nn.Linear)]
