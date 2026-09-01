"""Task-by-task checkpoint manager for Class-Incremental Learning.

Provides clean, atomic, task-level checkpointing:
- Saves ``task_{task_id:02d}.pt`` after each task.
- Atomic writes via a temporary file avoid corrupted checkpoints if interrupted.
- Automatically discovers the latest task checkpoint to resume execution seamlessly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch


class TaskCheckpointManager:
    """Manages clean task-by-task checkpoints for incremental learning runs."""

    FILENAME_PATTERNS = [
        re.compile(r"^task_(\d+)\.pt$"),
        re.compile(r"^checkpoint_task_(\d+)\.pt$"),
    ]

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_save(self, payload: dict[str, Any], target_path: Path) -> None:
        """Write to a temporary file first, then atomically rename to target."""
        tmp_path = target_path.with_suffix(".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(target_path)

    def task_path(self, task_id: int) -> Path:
        """Return canonical file path for task checkpoint: ``task_XX.pt``."""
        return self.output_dir / f"task_{task_id:02d}.pt"

    def save_task_checkpoint(self, task_id: int, payload: dict[str, Any]) -> Path:
        """Persist a task completion checkpoint atomically."""
        target = self.task_path(task_id)
        self._atomic_save(payload, target)
        return target

    def find_latest_checkpoint(self) -> tuple[int, Path | None]:
        """Scan directory and return (next_task_id, path_to_latest_task_checkpoint).

        Returns (0, None) if no task checkpoints exist.
        If task 2 was the last finished task, returns (3, Path('.../task_02.pt')).
        """
        latest_task = -1
        latest_path: Path | None = None

        if not self.output_dir.exists():
            return 0, None

        for file_path in sorted(self.output_dir.glob("*.pt")):
            for pattern in self.FILENAME_PATTERNS:
                match = pattern.match(file_path.name)
                if match:
                    task_id = int(match.group(1))
                    if task_id > latest_task:
                        latest_task = task_id
                        latest_path = file_path

        if latest_path is not None:
            return latest_task + 1, latest_path

        # Also support legacy checkpoint_boundary.pt if present
        boundary_file = self.output_dir / "checkpoint_boundary.pt"
        if boundary_file.exists():
            try:
                payload = torch.load(boundary_file, map_location="cpu", weights_only=False)
                finished_id = int(payload.get("progress", {}).get("task_id", 0))
                return finished_id + 1, boundary_file
            except Exception:
                pass

        return 0, None

    def list_task_checkpoints(self) -> list[tuple[int, Path]]:
        """Return list of (task_id, path) sorted by task_id."""
        results: list[tuple[int, Path]] = []
        if not self.output_dir.exists():
            return results

        for file_path in sorted(self.output_dir.glob("*.pt")):
            for pattern in self.FILENAME_PATTERNS:
                match = pattern.match(file_path.name)
                if match:
                    results.append((int(match.group(1)), file_path))
                    break

        results.sort(key=lambda item: item[0])
        return results

    @staticmethod
    def load_checkpoint(path: str | Path) -> dict[str, Any]:
        """Load checkpoint safely to CPU."""
        return torch.load(Path(path), map_location="cpu", weights_only=False)
