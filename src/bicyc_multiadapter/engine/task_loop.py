"""End-to-end Direction-1 runner: data -> model -> two-optimizer training -> CIL eval.

Checkpointing / resuming
------------------------
- ``checkpoint_boundary.pt``: rolling snapshot written after each finished task
  (older versions are overwritten, never accumulated).
- ``checkpoint_live.pt``: rolling mid-task snapshot (every ``N`` epochs and on
  KeyboardInterrupt) with optimizer + RNG state; deleted once the task finishes.
- Re-running the same command resumes automatically from the newest snapshot;
  ``history.jsonl`` and ``train_log.csv`` keep the full training/forgetting log.
"""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from bicyc_multiadapter.data.cil_dataset import CILDataManager, build_protocol
from bicyc_multiadapter.evaluation.metrics import summarize_accuracy_matrix
from bicyc_multiadapter.engine.keeplora_trainer import KeepLoRABiCycConfig, KeepLoRATrainer
from bicyc_multiadapter.models.alignment.bicyc import BidirectionalCycle
from bicyc_multiadapter.models.backbones.vit_timm import TimmViTEncoder
from bicyc_multiadapter.models.classifier import GaussianCILClassifier
from bicyc_multiadapter.models.keeplora_model import DEFAULT_TARGET_PATTERNS, KeepLoRACILModel
from bicyc_multiadapter.utils.reproducibility import enable_tf32, seed_everything


class DirectionOneExperiment:
    """Owns every component of one run and executes the full task sequence.

    Per task: gradient-SVD init -> epochs of two-optimizer training with online
    PFD updates -> Gaussian classifier refresh (transport old stats through A)
    -> end-of-task bookkeeping -> CIL evaluation on all seen tasks.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        experiment = cfg.experiment
        # Hydra >=1.1 packages secondary defaults relative to the containing group
        # unless declared with ``@_global_``, so model/data may live at the root or
        # under ``experiment`` depending on the config revision; accept both.
        model_cfg = cfg.model if "model" in cfg else experiment.model
        data_cfg = cfg.data if "data" in cfg else experiment.data
        self.device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
        if self.device == "cuda":
            enable_tf32()
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        seed_everything(int(experiment.seed), bool(cfg.deterministic))

        # Deterministic disjoint task split; class order persisted with results.
        self.protocol, self.class_order = build_protocol(
            int(data_cfg.num_classes),
            int(data_cfg.initial_classes),
            int(data_cfg.increment),
            int(data_cfg.class_order_seed),
        )
        keep_cfg, align_cfg = experiment.keeplora, experiment.alignment
        self.train_cfg = experiment.train
        self.data_manager = CILDataManager(
            root=str(data_cfg.root),
            protocol=self.protocol,
            image_size=int(experiment.image_size),
            batch_size=int(self.train_cfg.batch_size),
            num_workers=int(data_cfg.num_workers),
            base_seed=int(experiment.seed),
            pin_memory=self.device == "cuda",
        )

        # Frozen backbone + patched KeepLoRA layers + growing linear head.
        self.model = KeepLoRACILModel(
            encoder=TimmViTEncoder(str(model_cfg.backbone)),
            feature_dim=int(model_cfg.feature_dim),
            rank=int(model_cfg.lora_rank),
            alpha=float(model_cfg.lora_alpha),
            weight_energy=float(keep_cfg.weight_energy),
            feature_energy=float(keep_cfg.feature_energy),
            merge_after_task=bool(keep_cfg.merge_after_task),
            router_similarity=str(keep_cfg.router_similarity),
            router_temperature=float(keep_cfg.router_temperature),
            router_top_k=keep_cfg.router_top_k,
            target_patterns=tuple(experiment.get("targets", DEFAULT_TARGET_PATTERNS)),
            activation_cache_rows=int(experiment.get("activation_cache_rows", 4096)),
        )
        self.model.patch_backbone()
        self.model.to(self.device)

        self.align_config = KeepLoRABiCycConfig(
            lambda_bi=float(align_cfg.lambda_bi),
            lambda_cyc=float(align_cfg.lambda_cyc),
            lambda_min=float(align_cfg.lambda_min),
            lambda_max=float(align_cfg.lambda_max),
            distribution_temperature=float(align_cfg.distribution_temperature),
            use_adaptive_gate=bool(align_cfg.adaptive_gate),
            use_distillation=bool(align_cfg.enabled),
            anti_collapse_weight=float(align_cfg.get("anti_collapse_weight", 0.0)),
            use_amp=bool(self.train_cfg.get("amp", False)),
            amp_dtype=str(self.train_cfg.get("amp_dtype", "float16")),
        )
        self.bicycle = BidirectionalCycle(int(model_cfg.feature_dim)).to(self.device)
        self.classifier = GaussianCILClassifier(
            int(model_cfg.feature_dim),
            covariance_mode=str(experiment.classifier.covariance_mode),
            shrinkage=float(experiment.classifier.shrinkage),
        )
        self.old_model: KeepLoRACILModel | None = None
        self.accuracy_matrix: list[list[float]] = []
        # Rolling checkpoints + logs for resuming interrupted runs (cloud sessions).
        self.boundary_path = self.output_dir / "checkpoint_boundary.pt"
        self.live_path = self.output_dir / "checkpoint_live.pt"
        self.history_path = self.output_dir / "history.jsonl"
        self.train_log_path = self.output_dir / "train_log.csv"
        self.resume_enabled = bool(experiment.get("resume", True))
        self.checkpoint_every_epochs = int(experiment.get("checkpoint_every_epochs", 0))
        self._pending_optimizer_states: dict | None = None

    # ------------------------------------------------------------------- run
    def run(self) -> dict[str, float]:
        """Train every task in protocol order and persist checkpoint + metrics.

        If a previous interrupted run left checkpoints in ``output_dir``, training
        continues from there instead of starting over.
        """
        start_task, start_epoch, resumed_live = self._maybe_resume()
        if start_task or start_epoch or resumed_live:
            print(f"[resume] tiep tuc tu task {start_task}, epoch {start_epoch}")
        for spec in self.protocol.tasks[start_task:]:
            first_epoch = start_epoch if spec.task_id == start_task else 0
            skip_init = resumed_live and spec.task_id == start_task
            self._train_task(spec, first_epoch=first_epoch, skip_task_init=skip_init)
            row = self._evaluate_upto(spec.task_id)
            self._log_history(spec.task_id, row)
            print(f"[task {spec.task_id}] per-task accuracy: {[round(a, 4) for a in row]}")
            self._save_boundary_checkpoint(spec.task_id)
            self._write_running_metrics()
            if self.live_path.exists():
                self.live_path.unlink()  # boundary snapshot supersedes the mid-task one
        summary = summarize_accuracy_matrix(self.accuracy_matrix)
        self._save_outputs(summary)
        return summary

    def _train_task(self, spec, first_epoch: int = 0, skip_task_init: bool = False) -> None:
        train_loader, _ = self.data_manager.task_loaders(spec)
        task_id = spec.task_id
        total_epochs = int(self.train_cfg.epochs_per_task)
        resumed_mid_task = skip_task_init
        self.model.expand_head(spec.class_ids)
        if resumed_mid_task:
            # Adapter/PFD/head state arrives from the checkpoint; only re-arm hooks.
            self.model._register_hooks()
            print(f"[task {task_id}] resume giua task tu epoch {first_epoch}")
        else:
            # One CE-only backward pass gives G_t; residual-SVD init of A/B; hooks on.
            self.model.begin_task(task_id, train_loader, self.device)

        model_optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=float(self.train_cfg.lr),
            weight_decay=float(self.train_cfg.weight_decay),
        )
        alignment_optimizer = torch.optim.AdamW(
            self.bicycle.parameters(), lr=float(self.cfg.experiment.alignment.get("alignment_lr", self.train_cfg.lr))
        )
        if resumed_mid_task and self._pending_optimizer_states is not None:
            model_optimizer.load_state_dict(self._pending_optimizer_states["model"])
            alignment_optimizer.load_state_dict(self._pending_optimizer_states["alignment"])
            self._pending_optimizer_states = None
        trainer = KeepLoRATrainer(
            self.model, self.old_model, self.bicycle, model_optimizer, alignment_optimizer, self.align_config
        )
        self.model.train()
        epoch = first_epoch
        try:
            for epoch in range(first_epoch, total_epochs):
                running: dict[str, float] = {}
                for images, labels in tqdm(train_loader, desc=f"task {task_id} epoch {epoch}", leave=False):
                    stats = trainer.train_batch(images.to(self.device, non_blocking=True), labels.to(self.device))
                    self.model.update_routing_statistics(task_id)  # online PFD means
                    for key, value in stats.items():
                        running[key] = running.get(key, 0.0) + value / len(train_loader)
                for key, value in running.items():
                    self.writer.add_scalar(f"{key}/task{task_id}", value, epoch)
                self._append_epoch_log(task_id, epoch, running)
                print(f"[task {task_id} epoch {epoch}] " + " ".join(f"{k}={v:.4f}" for k, v in running.items()))
                next_epoch = epoch + 1
                if (
                    self.checkpoint_every_epochs > 0
                    and next_epoch % self.checkpoint_every_epochs == 0
                    and next_epoch < total_epochs
                ):
                    self._save_live_checkpoint(
                        task_id, next_epoch, model_optimizer, alignment_optimizer, next_epoch=next_epoch
                    )
        except KeyboardInterrupt:
            print(f"[interrupt] dung tai task {task_id} epoch {epoch}; luu checkpoint de resume...")
            self._save_live_checkpoint(task_id, epoch, model_optimizer, alignment_optimizer, next_epoch=epoch)
            raise

        # Consolidation: transport old class stats through A, then fit current-task stats.
        self.model.eval()
        features, labels = self._collect_features(train_loader)
        if task_id > 0:
            self.classifier.transport(self.bicycle.old_to_new)  # mu'=A(mu), Sigma'=A(Sigma)A^T
        self.classifier.fit_task(features, labels)
        self.model.end_task(task_id)
        self.old_model = self.model.snapshot()  # immutable teacher for the next task

    @torch.no_grad()
    def _collect_features(self, loader) -> tuple[torch.Tensor, torch.Tensor]:
        """Pooled backbone features for every sample; caller controls train/eval mode."""
        features, labels = [], []
        for images, targets in loader:
            _, batch_features = self.model(images.to(self.device, non_blocking=True))
            features.append(batch_features.cpu())
            labels.append(targets)
        return torch.cat(features), torch.cat(labels)

    @torch.no_grad()
    def _evaluate_upto(self, finished_task: int) -> list[float]:
        """CIL accuracy on all seen tasks with the Gaussian classifier (no task-id hint)."""
        self.model.eval()
        row: list[float] = []
        for spec in self.protocol.tasks[: finished_task + 1]:
            _, test_loader = self.data_manager.task_loaders(spec)
            features, labels = self._collect_features(test_loader)
            predictions = self.classifier.predict(features.to(self.device)).cpu()
            row.append(float((predictions == labels).float().mean()))
        self.accuracy_matrix.append(row)
        return row

    def _base_payload(self) -> dict:
        """Common checkpoint content; only compact statistics, never raw samples."""
        return {
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "model": self.model.state_dict(),
            "memory": self.model.memory_state_dict(),
            "bicycle": self.bicycle.state_dict(),
            "classifier": self.classifier.export_state(),
            "class_order": self.class_order,
            "accuracy_matrix": self.accuracy_matrix,
        }

    @staticmethod
    def _rng_state() -> dict:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng(state: dict) -> None:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def _atomic_save(self, payload: dict, target: Path) -> None:
        """Write via a temp file then rename, so a crash cannot corrupt the checkpoint."""
        tmp = target.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(target)

    def _save_boundary_checkpoint(self, finished_task: int) -> None:
        """Rolling snapshot after a finished task; replaces any older boundary."""
        payload = self._base_payload()
        payload["progress"] = {"status": "task_done", "task_id": finished_task}
        self._atomic_save(payload, self.boundary_path)

    def _save_live_checkpoint(
        self,
        task_id: int,
        completed_epochs: int,
        model_optimizer,
        alignment_optimizer,
        next_epoch: int | None = None,
    ) -> None:
        """Rolling mid-task snapshot (model + optimizers + RNG) for exact resume."""
        payload = self._base_payload()
        payload["progress"] = {
            "status": "training",
            "task_id": task_id,
            "next_epoch": int(next_epoch if next_epoch is not None else completed_epochs),
        }
        payload["model_optimizer"] = model_optimizer.state_dict()
        payload["alignment_optimizer"] = alignment_optimizer.state_dict()
        payload["rng"] = self._rng_state()
        self._atomic_save(payload, self.live_path)
    def _apply_state(self, payload: dict, with_snapshot: bool, with_optimizers: bool = False) -> None:
        """Restore one checkpoint payload into the live experiment objects."""
        if [int(c) for c in payload["class_order"]] != [int(c) for c in self.class_order]:
            raise ValueError("Checkpoint class order khong khop protocol hien tai; khong the resume.")
        self.model.trim_structure(payload["model"])
        self.model.restore_structure(payload["model"])
        self.model.load_state_dict(payload["model"])
        self.model.load_memory_state(payload.get("memory", {}))
        self.bicycle.load_state_dict(payload["bicycle"])
        self.classifier.load_state(payload["classifier"])
        self.accuracy_matrix = [list(map(float, row)) for row in payload["accuracy_matrix"]]
        if "rng" in payload:
            self._restore_rng(payload["rng"])
        if with_snapshot:
            # Teacher for the next task == model right after its end_task bookkeeping.
            self.old_model = self.model.snapshot()
        if with_optimizers:
            self._pending_optimizer_states = {
                "model": payload.get("model_optimizer"),
                "alignment": payload.get("alignment_optimizer"),
            }

    def _maybe_resume(self) -> tuple[int, int, bool]:
        """Return ``(task_id, first_epoch, resumed_from_live_snapshot)``."""
        if not self.resume_enabled:
            return 0, 0, False
        if self.live_path.exists():
            # Mid-task resume needs the previous boundary as the frozen teacher.
            if self.boundary_path.exists():
                boundary = torch.load(self.boundary_path, map_location="cpu", weights_only=False)
                self._apply_state(boundary, with_snapshot=True)
            payload = torch.load(self.live_path, map_location="cpu", weights_only=False)
            self._apply_state(payload, with_snapshot=False, with_optimizers=True)
            progress = payload["progress"]
            return int(progress["task_id"]), int(progress["next_epoch"]), True
        if self.boundary_path.exists():
            payload = torch.load(self.boundary_path, map_location="cpu", weights_only=False)
            self._apply_state(payload, with_snapshot=True)
            return int(payload["progress"]["task_id"]) + 1, 0, False
        return 0, 0, False

    def _log_history(self, finished_task: int, row: list[float]) -> None:
        """Append one forgetting-tracking record per finished task (survives crashes)."""
        summary_so_far = summarize_accuracy_matrix(self.accuracy_matrix)
        record = {
            "task_id": finished_task,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "per_task_accuracy": [round(float(a), 6) for a in row],
            **{f"running_{key}": round(float(value), 6) for key, value in summary_so_far.items()},
        }
        with open(self.history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.writer.add_scalar("eval/running_last_average", summary_so_far["last_average"], finished_task)
        self.writer.add_scalar(
            "eval/running_incremental_average", summary_so_far["incremental_average"], finished_task
        )
        self.writer.add_scalar("eval/running_forgetting", summary_so_far["forgetting"], finished_task)

    _EPOCH_LOG_KEYS = (
        "loss/ce",
        "loss/model",
        "loss/backward",
        "loss/alignment",
        "distribution/distance",
        "distribution/lambda_adaptive",
    )

    def _append_epoch_log(self, task_id: int, epoch: int, running: dict[str, float]) -> None:
        """Append per-epoch losses to ``train_log.csv`` (append-safe across resumes)."""
        new_file = not self.train_log_path.exists()
        with open(self.train_log_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(["task_id", "epoch", *self._EPOCH_LOG_KEYS])
            values = [round(running.get(key, float("nan")), 6) for key in self._EPOCH_LOG_KEYS]
            writer.writerow([task_id, epoch, *values])

    def _write_running_metrics(self, summary: dict[str, float] | None = None) -> None:
        """Rewrite metrics.json + accuracy CSV after every finished task."""
        if summary is None:
            summary = summarize_accuracy_matrix(self.accuracy_matrix)
        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "accuracy_matrix": self.accuracy_matrix}, handle, indent=2)
        with open(self.output_dir / "accuracy_matrix.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(self.accuracy_matrix)

    def _save_outputs(self, summary: dict[str, float]) -> None:
        """Final checkpoint (bases/factors/classifier only), metrics JSON and accuracy CSV."""
        torch.save(self._base_payload(), self.output_dir / "checkpoint_last.pt")
        self._write_running_metrics(summary)
        self.writer.add_hparams(
            {"experiment": str(self.cfg.experiment.name), "seed": str(self.cfg.experiment.seed)},
            {key: float(value) for key, value in summary.items()},
        )
        self.writer.close()

    @torch.no_grad()
    def evaluate_final_from_checkpoint(self, payload: dict) -> dict[str, float]:
        """Reload states from a checkpoint payload and re-score the final model once."""
        self.model.restore_structure(payload["model"])
        self.model.load_state_dict(payload["model"])
        self.model.load_memory_state(payload.get("memory", {}))
        self.bicycle.load_state_dict(payload["bicycle"])
        self.classifier.load_state(payload["classifier"])
        self.model.eval()
        final_row: list[float] = []
        for spec in self.protocol.tasks:
            _, test_loader = self.data_manager.task_loaders(spec)
            features, labels = self._collect_features(test_loader)
            predictions = self.classifier.predict(features.to(self.device)).cpu()
            final_row.append(float((predictions == labels).float().mean()))
        return {"final_per_task": final_row, "final_average": sum(final_row) / len(final_row)}
