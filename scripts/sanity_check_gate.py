"""Temporary sanity check: run a tiny 2-task experiment and verify the new logs."""
import json
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn

from bicyc_multiadapter.engine import task_loop as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "unit"))
from test_resume import FakeDataManager, TinyEncoder  # noqa: E402


class TinyEncoderDebug(TinyEncoder):
    def forward_features(self, images):
        out = super().forward_features(images)
        return out


def build_cfg(tmp_path) -> DictConfig:
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
                "name": "sanity",
                "seed": 7,
                "trainer": "keeplora",
                "resume": False,
                "checkpoint_every_epochs": 0,
                "image_size": 8,
                "targets": ["qkv", "fc1"],
                "activation_cache_rows": 32,
                "train": {
                    "epochs_per_task": 1,
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


with tempfile.TemporaryDirectory() as tmp:
    tl.TimmViTEncoder = lambda name, pretrained=True: TinyEncoder()  # type: ignore[attr-defined]
    tl.CILDataManager = FakeDataManager  # type: ignore[attr-defined]
    experiment = tl.DirectionOneExperiment(build_cfg(Path(tmp)))
    experiment.run()

    csv_path = experiment.output_dir / "train_log.csv"
    print("=== train_log.csv header ===")
    print(csv_path.read_text(encoding="utf-8").splitlines()[0])

    log_text = (experiment.output_dir / "run.log").read_text(encoding="utf-8")
    print("=== [routing] lines ===")
    for line in log_text.splitlines():
        if "[routing]" in line:
            print(line)

    # Direct probe after the run: inspect last_input and routing weights per layer.
    model = experiment.model
    spec = experiment.protocol.tasks[0]
    _, test_loader = experiment.data_manager.task_loaders(spec)
    images, _ = test_loader[0]
    print("=== direct routing_probe ===")
    print("model device:", next(model.parameters()).device)
    print("layers:", list(model.layers.keys()))
    print("memory keys:", list(model.memory.keys()))
    report = model.routing_probe(images)
    print("report:", report)
    for name, layer in model.layers.items():
        last = model.memory[name].last_input
        print(f"layer={name} last_input={None if last is None else tuple(last.shape)}")
        if last is not None:
            print("  route_report:", layer.route_report(last))

