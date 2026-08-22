from __future__ import annotations

from torch import nn


class RSIATTrainer:
    """Trainer for one shared adapter and training-only Bi-RAE."""

    def __init__(self, model: nn.Module, bi_rae: nn.Module) -> None:
        self.model, self.bi_rae = model, bi_rae

    def train_task(self, task_id: int) -> None:
        raise NotImplementedError("Implement CE + RS/orthogonal + bidirectional alignment.")
