"""Unit tests for TaskCheckpointManager."""

import torch

from bicyc_multiadapter.utils.checkpoint import TaskCheckpointManager


def test_task_checkpoint_manager_lifecycle(tmp_path) -> None:
    manager = TaskCheckpointManager(tmp_path / "checkpoints")

    # Initially empty
    next_task, latest_path = manager.find_latest_checkpoint()
    assert next_task == 0
    assert latest_path is None
    assert manager.list_task_checkpoints() == []

    # Save task 0
    payload_0 = {"task_id": 0, "progress": {"task_id": 0, "status": "task_done"}, "weight": torch.tensor([1.0, 2.0])}
    saved_0 = manager.save_task_checkpoint(0, payload_0)
    assert saved_0.name == "task_00.pt"
    assert saved_0.exists()

    next_task, latest_path = manager.find_latest_checkpoint()
    assert next_task == 1
    assert latest_path == saved_0

    # Save task 1
    payload_1 = {"task_id": 1, "progress": {"task_id": 1, "status": "task_done"}, "weight": torch.tensor([3.0, 4.0])}
    saved_1 = manager.save_task_checkpoint(1, payload_1)
    assert saved_1.name == "task_01.pt"
    assert saved_1.exists()

    next_task, latest_path = manager.find_latest_checkpoint()
    assert next_task == 2
    assert latest_path == saved_1

    # Check list
    checkpoints = manager.list_task_checkpoints()
    assert len(checkpoints) == 2
    assert checkpoints[0] == (0, saved_0)
    assert checkpoints[1] == (1, saved_1)

    # Load checkpoint
    loaded = manager.load_checkpoint(saved_1)
    assert loaded["task_id"] == 1
    assert torch.allclose(loaded["weight"], torch.tensor([3.0, 4.0]))
