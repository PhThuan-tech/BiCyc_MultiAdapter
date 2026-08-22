"""Dataset and class-incremental task split interfaces."""

from .cil_dataset import CILDataManager, build_protocol
from .task_protocol import ClassIncrementalProtocol, TaskSpec

__all__ = ["CILDataManager", "ClassIncrementalProtocol", "TaskSpec", "build_protocol"]
