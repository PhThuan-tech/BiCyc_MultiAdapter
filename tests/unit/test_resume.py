"""Integration tests: rolling checkpoints, automatic resume, forgetting logs."""

import json

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from bicyc_multiadapter.engine import task_loop as tl


class TinyBlock(nn.Module):
    """MLP block whose projections carry KeepLoRA target names."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(8, 16)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.gelu(self.attn.qkv(x))
        return self.mlp.fc1(x)


class TinyEncoder(nn.Module):
    """Stand-in for TimmViTEncoder: frozen, exposes ``network`` + ``forward_features``."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(TinyBlock())
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class FakeDataManager:
    """Keyword-compatible CILDataManager stub serving small synthetic loaders."""

    def __init__(self, root, protocol, image_size, batch_size, num_workers, base_seed, pin_memory=False):
        self.protocol = protocol
        self.base_seed = base_seed

    def _loader(self, spec, seed_offset: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        generator = torch.Generator().manual_seed(self.base_seed + spec.task_id + seed_offset)
        batches = []
        for _ in range(3):  # 3 batches/epoch keeps the test fast
            images = torch.randn(4, 8, generator=generator)
            labels = torch.tensor([spec.class_ids[i % len(spec.class_ids)] for i in range(4)])
            batches.append((images, labels))
        return batches

    def task_loaders(self, spec):
        return self._loader(spec, 100), self._loader(spec, 0)


def build_cfg(tmp_path, epochs_per_task: int, checkpoint_every: int) -> DictConfig:
    return DictConfig(
        {
            "device": "cpu",
            "deterministic": False,
            "output_dir": str(tmp_path / "run"),
            "model": {"backbone": "tiny", "feature_dim": 8, "lora_rank": 2, "lora_alpha": 4},
            "data": {
                "root": "unused",
                "num_classes": 20,
                "initial_classes": 10,
                "increment": 10,
                "class_order_seed": 1993,
                "num_workers": 0,
            },
            "experiment": {
                "name": "resume_test",
                "seed": 7,
                "trainer": "keeplora",
                "resume": True,
                "checkpoint_every_epochs": checkpoint_every,
                "image_size": 8,
                "targets": ["qkv", "fc1"],
                "activation_cache_rows": 32,
                "train": {
                    "epochs_per_task": epochs_per_task,
                    "batch_size": 4,
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                    "amp": False,
                    "amp_dtype": "float16",
                },
                "alignment": {
                    "enabled": True,
                    "adaptive_gate": True,
                    "lambda_bi": 1.0,
                    "lambda_cyc": 1.0,
                    "anti_collapse_weight": 0.0,
                    "lambda_min": 0.1,
                    "lambda_max": 1.0,
                    "distribution_temperature": 1.0,
                    "alignment_lr": 1e-3,
                },
                "keeplora": {
                    "weight_energy": 0.9,
                    "feature_energy": 0.9,
                    "merge_after_task": False,
                    "router_similarity": "l2",
                    "router_temperature": 1.0,
                    "router_top_k": None,
                },
                "classifier": {"covariance_mode": "diagonal", "shrinkage": 1e-4},
            },
        }
    )


def make_experiment(monkeypatch, tmp_path, epochs: int = 2, every: int = 0) -> tl.DirectionOneExperiment:
    monkeypatch.setattr(tl, "TimmViTEncoder", lambda name, pretrained=True: TinyEncoder())
    monkeypatch.setattr(tl, "CILDataManager", FakeDataManager)
    return tl.DirectionOneExperiment(build_cfg(tmp_path, epochs, every))


def test_full_run_writes_logs_and_boundary_resume(monkeypatch, tmp_path) -> None:
    experiment = make_experiment(monkeypatch, tmp_path)
    summary = experiment.run()
    out_dir = experiment.output_dir

    assert (out_dir / "checkpoint_boundary.pt").exists()
    assert (out_dir / "checkpoint_last.pt").exists()
    history_lines = (out_dir / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(history_lines) == 2  # one record per finished task
    first_record = json.loads(history_lines[0])
    assert first_record["task_id"] == 0 and "running_forgetting" in first_record
    log_rows = (out_dir / "train_log.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(log_rows) == 1 + 2 * 2  # header + epochs x tasks
    assert set(summary) == {"last_average", "incremental_average", "forgetting"}

    # A brand-new experiment must pick up the finished run from the boundary file.
    resumed = make_experiment(monkeypatch, tmp_path)
    assert resumed._maybe_resume() == (2, 0, False)
    assert resumed.accuracy_matrix == experiment.accuracy_matrix
    assert resumed.old_model is not None
    assert any(
        memory.feature_basis is not None for memory in resumed.model.memory.values()
    )  # protected bases (W_p, M_t) restored for the next task's SVD


def test_interrupted_run_resumes_from_live_checkpoint(monkeypatch, tmp_path) -> None:
    experiment = make_experiment(monkeypatch, tmp_path, epochs=2, every=1)

    original_step = tl.KeepLoRATrainer.train_batch
    calls = {"count": 0}

    def flaky_step(self, images, labels):
        calls["count"] += 1
        if calls["count"] >= 9:  # task 0 needs 6 steps; die early inside task 1
            raise KeyboardInterrupt
        return original_step(self, images, labels)

    tl.KeepLoRATrainer.train_batch = flaky_step
    try:
        with pytest.raises(KeyboardInterrupt):
            experiment.run()
    finally:
        tl.KeepLoRATrainer.train_batch = original_step

    assert experiment.live_path.exists()
    assert experiment.boundary_path.exists()

    resumed = make_experiment(monkeypatch, tmp_path, epochs=2, every=1)
    start_task, start_epoch, resumed_live = resumed._maybe_resume()
    assert (start_task, start_epoch, resumed_live) == (1, 0, True)
    assert resumed.old_model is not None  # teacher rebuilt from the boundary snapshot
    assert resumed.accuracy_matrix == experiment.accuracy_matrix  # task 0 row kept

    summary = resumed.run()  # finishes task 1 and writes all outputs
    assert len(resumed.accuracy_matrix) == 2
    history_lines = (resumed.output_dir / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(history_lines) == 2
    assert json.loads(history_lines[-1])["task_id"] == 1
    assert not resumed.live_path.exists()  # superseded by the final boundary
    assert 0.0 <= summary["last_average"] <= 1.0
