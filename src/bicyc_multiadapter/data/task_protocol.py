from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    class_ids: tuple[int, ...]


class ClassIncrementalProtocol:
    """Deterministic task order. Persist this manifest with each experiment."""

    def __init__(self, tasks: tuple[TaskSpec, ...], exemplar_free: bool = True) -> None:
        self.tasks = tasks
        self.exemplar_free = exemplar_free
