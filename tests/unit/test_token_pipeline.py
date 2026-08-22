"""Regression test: real ViT linears receive [batch, tokens, dim] activations.

The forward hook, PFD routing statistics and the end-of-task SVD must all work
on flattened token rows instead of raising on 3-D tensors.
"""

import torch
from torch import nn

from bicyc_multiadapter.engine.keeplora_trainer import KeepLoRABiCycConfig, KeepLoRATrainer
from bicyc_multiadapter.models.alignment.bicyc import BidirectionalCycle
from bicyc_multiadapter.models.classifier import GaussianCILClassifier
from bicyc_multiadapter.models.keeplora_model import KeepLoRACILModel


class TokenBlock(nn.Module):
    """Tiny transformer-ish block whose linears get token-sequence inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(8, 16)
        self.mlp.fc2 = nn.Linear(16, 8)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # [B, T, 8]
        tokens = torch.nn.functional.gelu(self.mlp.fc1(tokens))
        return self.mlp.fc2(tokens)


class TokenEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = TokenBlock()

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images).mean(dim=1)


def _make_model(activation_cache_rows: int = 64) -> KeepLoRACILModel:
    encoder = TokenEncoder().eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    model = KeepLoRACILModel(
        encoder, feature_dim=8, rank=2, alpha=4, activation_cache_rows=activation_cache_rows
    )
    model.patch_backbone()
    return model


def _token_loader() -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    return [
        (
            torch.randn(4, 5, 8, generator=generator),
            torch.randint(0, 10, (4,), generator=generator),
        )
        for _ in range(3)
    ]


def test_hook_flattens_token_activations_and_caps_rows() -> None:
    torch.manual_seed(0)
    model = _make_model()
    loader = _token_loader()
    model.expand_head(tuple(range(10)))
    model.begin_task(0, loader, "cpu")
    model.train()
    for images, labels in loader:
        logits, features = model(images)
        assert logits.shape == (4, 10)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        model.update_routing_statistics(0)  # must not raise on 3-D layer inputs
    memory = model.memory["mlp-fc1"]
    # Rows are counted per token row and capped at activation_cache_rows.
    assert memory.cached_rows <= 64
    assert all(chunk.ndim == 2 for chunk in memory.activation_cache)

    classifier = GaussianCILClassifier(8)
    model.eval()
    features, labels = zip(*[(model(images)[1], lbls) for images, lbls in loader])
    classifier.fit_task(torch.cat(features).detach(), torch.cat(labels))
    model.end_task(0)  # end-of-task SVD must not raise on token rows
    assert model.memory["mlp-fc1"].feature_basis is not None


def test_trainer_supports_amp_config_on_cpu() -> None:
    """AMP is requested but silently disabled off-CUDA; the step must still run."""
    torch.manual_seed(0)
    model = _make_model()
    loader = _token_loader()
    model.expand_head(tuple(range(10)))
    model.begin_task(0, loader, "cpu")
    bicycle = BidirectionalCycle(8)
    config = KeepLoRABiCycConfig(use_amp=True, amp_dtype="float16")
    trainer = KeepLoRATrainer(
        model,
        None,
        bicycle,
        torch.optim.AdamW(model.trainable_parameters(), lr=1e-3),
        torch.optim.AdamW(bicycle.parameters(), lr=1e-3),
        config,
    )
    assert not trainer.amp_enabled
    images, labels = loader[0]
    stats = trainer.train_batch(images, labels)
    assert "loss/ce" in stats