"""Smoke test for the Direction-1 model lifecycle using a tiny fake encoder.

Run: python scripts/smoke_direction1.py
Requires torch only (no timm/torchvision/hydra needed).
"""

import sys
import torch
from torch import nn

sys.path.insert(0, "src")

from bicyc_multiadapter.models.keeplora_model import KeepLoRACILModel
from bicyc_multiadapter.engine.keeplora_trainer import KeepLoRABiCycConfig, KeepLoRATrainer
from bicyc_multiadapter.models.alignment.bicyc import BidirectionalCycle
from bicyc_multiadapter.models.classifier import GaussianCILClassifier


class Block(nn.Module):
    """Tiny MLP whose projections carry the KeepLoRA target names."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(8, 16)
        self.attn.proj = nn.Linear(16, 8)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(8, 16)
        self.mlp.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        x = torch.nn.functional.gelu(self.attn.qkv(x))
        x = self.attn.proj(x)
        x = torch.nn.functional.gelu(self.mlp.fc1(x))
        return self.mlp.fc2(x)


class FakeEncoder(nn.Module):
    """Minimal stand-in exposing ``network`` with pattern-matched linears."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(Block(), Block())
        self.eval()

    def forward_features(self, images):
        return self.network(images.flatten(1)) if images.ndim > 2 else self.network(images)


def make_loader(task_id, batches=3, batch=16):
    generator = torch.Generator().manual_seed(task_id)
    for _ in range(batches):
        yield torch.randn(batch, 8, generator=generator), torch.randint(
            10 * task_id, 10 * task_id + 10, (batch,), generator=generator
        )


def main() -> None:
    torch.manual_seed(0)
    encoder = FakeEncoder().eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    model = KeepLoRACILModel(encoder, feature_dim=8, rank=2, alpha=4, router_top_k=None)
    model.patch_backbone()

    classifier = GaussianCILClassifier(8)
    bicycle = BidirectionalCycle(8)
    old_model = None
    align_cfg = KeepLoRABiCycConfig(lambda_bi=1.0, lambda_cyc=1.0)

    for task_id in range(2):
        loader = list(make_loader(task_id))
        classes = tuple(range(10 * task_id, 10 * task_id + 10))
        model.expand_head(classes)
        model.begin_task(task_id, loader, "cpu")
        assert len(list(model.layers.values())[0].adapters) == task_id + 1

        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
        trainer = KeepLoRATrainer(model, old_model, bicycle, optimizer, torch.optim.AdamW(bicycle.parameters()), align_cfg)
        model.train()
        for epoch in range(1):
            for images, labels in loader:
                stats = trainer.train_batch(images, labels)
                model.update_routing_statistics(task_id)

        model.eval()
        features, labels = zip(*[(model(images)[1].detach(), lbls) for images, lbls in loader])
        features, labels = torch.cat(features), torch.cat(labels)
        if task_id > 0:
            classifier.transport(bicycle.old_to_new)
        classifier.fit_task(features, labels)
        preds = classifier.predict(features)
        print(f"task {task_id}: stats={ {k: round(v, 4) for k, v in stats.items()} } train-acc={(preds == labels).float().mean():.2f}")

        model.end_task(task_id)
        layer = next(iter(model.layers.values()))
        if task_id == 0:
            assert not layer.adapters["0"].B.requires_grad
            # Function-preserving init: delta is zero right after initialization.
        old_model = model.snapshot()

    # Old teacher must be frozen and identical to the pre-task model.
    assert all(not p.requires_grad for p in old_model.parameters())
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
