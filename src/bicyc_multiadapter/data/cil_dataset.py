"""Exemplar-free CIL data protocol for CIFAR-100.

Builds a deterministic class order from ``class_order_seed``, slices it into
disjoint tasks and serves per-task loaders. Raw exemplars of past tasks are
never retained anywhere in this module.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .task_protocol import ClassIncrementalProtocol, TaskSpec

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    """ViT preprocessing: upscale CIFAR's 32px frames and normalize with ImageNet stats."""
    train = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    test = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train, test


def build_protocol(
    num_classes: int, initial_classes: int, increment: int, class_order_seed: int
) -> tuple[ClassIncrementalProtocol, list[int]]:
    """Deterministic disjoint task split; persist the returned class order with results."""
    generator = np.random.default_rng(class_order_seed)
    order = [int(label) for label in generator.permutation(num_classes)]
    sizes = [initial_classes] + [increment] * ((num_classes - initial_classes) // increment)
    specs, cursor = [], 0
    for task_id, size in enumerate(sizes):
        specs.append(TaskSpec(task_id=task_id, class_ids=tuple(order[cursor : cursor + size])))
        cursor += size
    return ClassIncrementalProtocol(tuple(specs)), order


class _TaskView(Dataset):
    """Read-only view of one CIFAR split restricted to a single task's classes."""

    def __init__(self, base: datasets.CIFAR100, class_ids: tuple[int, ...], transform) -> None:
        wanted = set(class_ids)
        self.base, self.transform = base, transform
        self.indices = [index for index, label in enumerate(base.targets) if label in wanted]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source = self.indices[index]
        image = Image.fromarray(self.base.data[source])
        return self.transform(image), int(self.base.targets[source])


class CILDataManager:
    """Downloads CIFAR-100 once and yields per-task loaders in protocol order."""

    def __init__(
        self,
        root: str,
        protocol: ClassIncrementalProtocol,
        image_size: int,
        batch_size: int,
        num_workers: int,
        base_seed: int,
        pin_memory: bool = False,
        eval_batch_size: int | None = None,
    ) -> None:
        self.protocol = protocol
        self.batch_size = batch_size
        self.eval_batch_size = int(eval_batch_size) if eval_batch_size is not None else batch_size
        self.num_workers, self.base_seed = num_workers, base_seed
        self.pin_memory = pin_memory
        self.train_transform, self.test_transform = build_transforms(image_size)
        # download=True keeps data under ``root`` (host volume); nothing goes into git/image.
        self.train_split = datasets.CIFAR100(root, train=True, download=True)
        self.test_split = datasets.CIFAR100(root, train=False, download=True)

    def task_loaders(self, spec: TaskSpec) -> tuple[DataLoader, DataLoader]:
        """Train loader is shuffled with a task-seeded generator for reproducibility."""
        train_view = _TaskView(self.train_split, spec.class_ids, self.train_transform)
        test_view = _TaskView(self.test_split, spec.class_ids, self.test_transform)
        generator = torch.Generator().manual_seed(self.base_seed + spec.task_id)
        train_options = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # Persistent workers avoid re-forking processes for every task/evaluation.
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
        test_options = dict(
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
        train = DataLoader(train_view, shuffle=True, generator=generator, **train_options)
        test = DataLoader(test_view, shuffle=False, **test_options)
        return train, test

