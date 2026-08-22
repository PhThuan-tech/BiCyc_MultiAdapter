from __future__ import annotations

import torch
from torch import Tensor


def forgetting(best_accuracy: Tensor, current_accuracy: Tensor) -> Tensor:
    """Per-task forgetting: max historical accuracy minus current accuracy."""
    return (best_accuracy - current_accuracy).clamp_min(0)


def representation_drift(old_features: Tensor, new_features: Tensor) -> dict[str, Tensor]:
    """Report L2 and cosine distances for the same probe samples."""
    old = torch.nn.functional.normalize(old_features, dim=-1)
    new = torch.nn.functional.normalize(new_features, dim=-1)
    return {
        "l2": torch.linalg.vector_norm(old_features - new_features, dim=-1).mean(),
        "cosine": (1 - (old * new).sum(dim=-1)).mean(),
    }


def summarize_accuracy_matrix(matrix: list[list[float]]) -> dict[str, float]:
    """Standard CIL summary from the upper-triangular accuracy matrix.

    - ``last_average``: mean accuracy over all seen classes after the final task.
    - ``incremental_average``: mean over tasks t of the accuracy after task t.
    - ``forgetting``: mean over old tasks of (best past accuracy - final accuracy).
    """
    if not matrix:
        raise ValueError("The accuracy matrix is empty.")
    size = len(matrix)
    width = max(len(row) for row in matrix)
    padded = torch.full((size, width), float("nan"))
    for row_index, row in enumerate(matrix):
        padded[row_index, : len(row)] = torch.tensor(row, dtype=torch.float32)
    last_average = float(torch.nanmean(padded[-1]))
    lower_triangle = torch.tril(padded)
    incremental_average = float(torch.nanmean(lower_triangle))
    forgetting_per_task = []
    for column in range(width - 1):  # old tasks only; the final task cannot be forgotten
        history = padded[:-1, column]
        valid = history[~torch.isnan(history)]
        if valid.numel() == 0:
            continue
        best_past = float(valid.max())
        forgetting_per_task.append(max(0.0, best_past - float(padded[-1, column])))
    average_forgetting = sum(forgetting_per_task) / max(len(forgetting_per_task), 1)
    return {
        "last_average": last_average,
        "incremental_average": incremental_average,
        "forgetting": average_forgetting,
    }
