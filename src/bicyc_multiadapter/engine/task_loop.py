"""End-to-end Direction-1 runner: data -> model -> two-optimizer training -> CIL eval.

Checkpointing / resuming
------------------------
- ``checkpoint_boundary.pt``: rolling snapshot written after each finished task
  (older versions are overwritten, never accumulated). With
  ``experiment.keep_task_checkpoints=true`` an extra ``checkpoint_task_<t>.pt``
  is kept for every task.
- ``checkpoint_live.pt``: rolling mid-task snapshot (every ``N`` epochs and on
  KeyboardInterrupt) with optimizer + RNG state; deleted once the task finishes.
- All checkpoints are CPU-safe (every tensor is moved to CPU before
  serialisation) and carry ``checkpoint_version`` for compatibility checks on
  resume.
- Re-running the same command resumes automatically from the newest snapshot;
  ``history.jsonl`` and ``train_log.csv`` keep the full training/forgetting log.
- ``run.log`` (console + file logs) and ``run_meta.json`` (environment, resolved
  config, git revision) document every run.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    DictConfig = dict  # type: ignore[assignment,misc]
    OmegaConf = None  # type: ignore[assignment]
try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, AttributeError, ModuleNotFoundError):
    class SummaryWriter:  # type: ignore[no-redef]
        """Fallback mock when tensorboard is not installed."""
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def add_hparams(self, *args, **kwargs):
            pass
        def close(self):
            pass

from tqdm import tqdm

from bicyc_multiadapter.data.cil_dataset import CILDataManager, build_protocol
from bicyc_multiadapter.evaluation.metrics import summarize_accuracy_matrix
from bicyc_multiadapter.engine.keeplora_trainer import KeepLoRABiCycConfig, KeepLoRATrainer
from bicyc_multiadapter.models.alignment.bicyc import BidirectionalCycle
from bicyc_multiadapter.models.backbones.vit_timm import TimmViTEncoder
from bicyc_multiadapter.models.classifier import GaussianCILClassifier
from bicyc_multiadapter.models.keeplora_model import DEFAULT_TARGET_PATTERNS, KeepLoRACILModel
from bicyc_multiadapter.utils.checkpoint import TaskCheckpointManager
from bicyc_multiadapter.utils.logging_utils import collect_environment, setup_logging

from bicyc_multiadapter.utils.reproducibility import enable_tf32, seed_everything

# Payload schema version; bump whenever checkpoint fields change incompatibly.
# Files saved before versioning are treated as legacy v1 (structurally compatible).
CHECKPOINT_VERSION = 2


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
        eval_batch_size = int(experiment.get("eval_batch_size", max(int(self.train_cfg.batch_size) * 2, 128)))

        is_cuda = self.device == "cuda" or (isinstance(self.device, str) and self.device.startswith("cuda")) or (isinstance(self.device, torch.device) and self.device.type == "cuda")
        data_manager_kwargs = {
            "root": str(data_cfg.root),
            "protocol": self.protocol,
            "image_size": int(experiment.image_size),
            "batch_size": int(self.train_cfg.batch_size),
            "num_workers": int(data_cfg.num_workers),
            "base_seed": int(experiment.seed),
            "pin_memory": bool(is_cuda),
        }
        try:
            self.data_manager = CILDataManager(eval_batch_size=eval_batch_size, **data_manager_kwargs)
        except TypeError:
            self.data_manager = CILDataManager(**data_manager_kwargs)


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
            lambda_iso=float(align_cfg.get("lambda_iso", 0.1)),
            lambda_min=float(align_cfg.lambda_min),
            lambda_max=float(align_cfg.lambda_max),
            distribution_temperature=float(align_cfg.distribution_temperature),
            use_adaptive_gate=bool(align_cfg.adaptive_gate),
            channelwise_gate=bool(align_cfg.get("channelwise_gate", True)),
            use_distillation=bool(align_cfg.enabled),
            anti_collapse_weight=float(align_cfg.get("anti_collapse_weight", 0.0)),
            use_amp=bool(self.train_cfg.get("amp", False)),
            amp_dtype=str(self.train_cfg.get("amp_dtype", "float16")),
            phase_stability_start=float(self.train_cfg.get("phase_stability_start", 0.35)),
            phase_stability_min=float(self.train_cfg.get("phase_stability_min", 0.75)),
            phase_stability_max=float(self.train_cfg.get("phase_stability_max", 1.0)),
        )
        self.bicycle = BidirectionalCycle(int(model_cfg.feature_dim)).to(self.device)
        self.classifier = GaussianCILClassifier(
            int(model_cfg.feature_dim),
            covariance_mode=str(experiment.classifier.covariance_mode),
            shrinkage=float(experiment.classifier.shrinkage),
        )
        self.old_model: KeepLoRACILModel | None = None
        self.accuracy_matrix: list[list[float]] = []
        # Task-by-task checkpoint manager + rolling fallback paths
        self.checkpoint_manager = TaskCheckpointManager(self.output_dir)
        self.boundary_path = self.output_dir / "checkpoint_boundary.pt"
        self.last_path = self.output_dir / "checkpoint_last.pt"
        self.live_path = self.output_dir / "checkpoint_live.pt"
        self.history_path = self.output_dir / "history.jsonl"
        self.train_log_path = self.output_dir / "train_log.csv"
        self.resume_enabled = bool(experiment.get("resume", True))
        self.checkpoint_every_epochs = int(experiment.get("checkpoint_every_epochs", 0))
        self.keep_task_checkpoints = bool(experiment.get("keep_task_checkpoints", True))
        self._pending_optimizer_states: dict | None = None


        # Structured logging + run metadata (environment, config, git revision).
        self.log = setup_logging("bicyc", self.output_dir)
        self.run_meta_path = self.output_dir / "run_meta.json"
        self._write_run_meta()
        self._task_durations: list[float] = []
        self._run_peak_gib: float = 0.0
        self._last_task_train_stats: dict[str, float] | None = None

    # ------------------------------------------------------------------- run
    def run(self) -> dict[str, float]:
        """Train every task in protocol order and persist checkpoint + metrics.

        If a previous interrupted run left checkpoints in ``output_dir``, training
        continues from there instead of starting over.
        """
        start_task, start_epoch, resumed_live = self._maybe_resume()
        if start_task or start_epoch or resumed_live:
            self.log.info(
                "Resume tu checkpoint: bat dau tu task %s, epoch %s%s",
                start_task,
                start_epoch,
                " (giua task)" if resumed_live else " (dau task)",
            )
        # A mid-task checkpoint whose next_epoch is already past the configured
        # epoch budget (e.g. user resumed with a smaller epochs_per_task) would
        # silently "complete" the task with zero training. Fail loudly instead.
        if resumed_live and start_epoch >= int(self.train_cfg.epochs_per_task):
            raise ValueError(
                f"Checkpoint live bat dau tu epoch {start_epoch} nhung epochs_per_task hien tai "
                f"la {int(self.train_cfg.epochs_per_task)} (nho hon start_epoch). Ban da giam "
                "epochs_per_task so voi lan chay truoc? Hay xoa checkpoint_live.pt (va "
                "checkpoint_boundary.pt neu muon bat dau lai), hoac khoi phuc lai epochs_per_task "
                "dung nhu lan chay truoc."
            )
        self._log_run_config()
        run_started = time.perf_counter()
        total_tasks = len(self.protocol.tasks)
        for spec in self.protocol.tasks[start_task:]:
            first_epoch = start_epoch if spec.task_id == start_task else 0
            skip_init = resumed_live and spec.task_id == start_task
            if torch.cuda.is_available() and self.device != "cpu":
                torch.cuda.reset_peak_memory_stats(self.device)  # per-task peak
            task_started = time.perf_counter()
            self._train_task(spec, first_epoch=first_epoch, skip_task_init=skip_init)
            task_duration = time.perf_counter() - task_started
            self._task_durations.append(task_duration)
            row = self._evaluate_upto(spec.task_id)
            self._log_history(spec.task_id, row)
            self._log_task_summary(spec.task_id, row, task_duration, run_started, total_tasks)
            self._save_boundary_checkpoint(spec.task_id)
            self._write_running_metrics()
            if self.live_path.exists():
                self.live_path.unlink()  # boundary snapshot supersedes the mid-task one
        summary = summarize_accuracy_matrix(self.accuracy_matrix)
        self._save_outputs(summary)
        self.log.info(
            "Run hoan tat. last_average=%.4f incremental_average=%.4f forgetting=%.4f"
            " | tong=%s%s",
            summary["last_average"],
            summary["incremental_average"],
            summary["forgetting"],
            self._format_duration(time.perf_counter() - run_started),
            self._gpu_mem_summary(),
        )
        return summary
    # ------------------------------------------------------- console summaries
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Human-readable duration: ``3h 5m`` / ``12m 34s`` / ``8.5s``."""
        seconds = float(max(seconds, 0.0))
        if seconds >= 3600:
            return f"{seconds / 3600:.1f}h {int(seconds % 3600 / 60)}m"
        if seconds >= 60:
            return f"{int(seconds / 60)}m {int(seconds % 60)}s"
        return f"{seconds:.1f}s"

    def _gpu_mem_summary(self) -> str:
        """Live/peak CUDA memory in GiB (empty string on CPU runs)."""
        if not torch.cuda.is_available() or self.device == "cpu":
            return ""
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        self._run_peak_gib = max(
            self._run_peak_gib, torch.cuda.max_memory_allocated(self.device) / 1024**3
        )
        return f" | GPU alloc={allocated:.2f}GiB peak={self._run_peak_gib:.2f}GiB"

    def _log_run_config(self) -> None:
        """Echo every parameter that affects results so ``run.log`` is self-contained."""
        experiment = self.cfg.experiment
        model_cfg = self.cfg.model if "model" in self.cfg else experiment.model
        data_cfg = self.cfg.data if "data" in self.cfg else experiment.data
        train_cfg, align_cfg = experiment.train, experiment.alignment
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        rows = [
            ("experiment", str(experiment.name)),
            ("seed", str(int(experiment.seed))),
            ("device", f"{self.device} ({gpu_name})"),
            ("backbone", f"{model_cfg.backbone}@{int(experiment.image_size)}"),
            ("batch_size", str(int(train_cfg.batch_size))),
            ("epochs_per_task", str(int(train_cfg.epochs_per_task))),
            (
                "lr / weight_decay",
                f"{float(train_cfg.get('lr', 0.001))} / {float(train_cfg.get('weight_decay', 0.0001))}",
            ),
            ("amp", f"{bool(train_cfg.get('amp', False))} ({train_cfg.get('amp_dtype', 'float16')})"),
            ("lora_rank / alpha", f"{int(model_cfg.lora_rank)} / {int(model_cfg.lora_alpha)}"),
            ("targets", ",".join(self.model.target_patterns)),
            (
                "alignment",
                f"lambda_bi={align_cfg.lambda_bi:.2f} lambda_cyc={align_cfg.lambda_cyc:.2f} "
                f"lambda[{align_cfg.lambda_min:.2f}..{align_cfg.lambda_max:.2f}] "
                f"adaptive_gate={bool(align_cfg.adaptive_gate)}",
            ),
            ("activation_cache_rows", str(self.model.activation_cache_rows)),
            ("resume", str(self.resume_enabled)),
            ("checkpoint_every_epochs", str(self.checkpoint_every_epochs)),
            ("keep_task_checkpoints", str(self.keep_task_checkpoints)),
            ("data_root", str(data_cfg.root)),
            ("num_workers", str(int(data_cfg.num_workers))),
            ("output_dir", str(self.output_dir)),
        ]
        width = max(len(key) for key, _ in rows)
        self.log.info("================= RUN CONFIG =================")
        for key, value in rows:
            self.log.info("  %-*s : %s", width, key, value)
        self.log.info("==============================================")

    def _log_task_summary(
        self,
        task_id: int,
        row: list[float],
        task_duration: float,
        run_started: float,
        total_tasks: int,
    ) -> None:
        """Per-task block: accuracy row, delta vs previous task, running CIL
        summary, wall-clock task/cumulative/ETA, and VRAM footprint."""
        summary = summarize_accuracy_matrix(self.accuracy_matrix)
        cumulative = time.perf_counter() - run_started
        avg_per_task = cumulative / (task_id + 1)
        eta = avg_per_task * (total_tasks - task_id - 1)
        comparison = ""
        if len(self.accuracy_matrix) >= 2:
            previous = self.accuracy_matrix[-2]
            deltas = []
            for old_index in range(len(previous)):
                delta = row[old_index] - previous[old_index]
                deltas.append(f"{'+' if delta >= 0 else ''}{delta:.4f}")
            comparison = " | delta_cu=" + ",".join(deltas)
        row_text = ", ".join(f"{a:.4f}" for a in row)
        self.log.info(
            "[task %s] acc= %s%s | last_avg=%.4f inc_avg=%.4f forget=%.4f"
            " | time=%s cum=%s eta=%s%s",
            task_id,
            row_text,
            comparison,
            summary["last_average"],
            summary["incremental_average"],
            summary["forgetting"],
            self._format_duration(task_duration),
            self._format_duration(cumulative),
            self._format_duration(eta),
            self._gpu_mem_summary(),
        )



    def _train_task(self, spec, first_epoch: int = 0, skip_task_init: bool = False) -> None:
        train_loader, _ = self.data_manager.task_loaders(spec)
        task_id = spec.task_id
        total_epochs = int(self.train_cfg.epochs_per_task)
        resumed_mid_task = skip_task_init
        self.model.expand_head(spec.class_ids)
        self.model.head.to(self.device)  # head is created lazily after .to(device)
        if resumed_mid_task:
            # Adapter/PFD/head state arrives from the checkpoint; only re-arm hooks.
            self.model._register_hooks()
            self.log.info("Resume giua task %s tu epoch %s", task_id, first_epoch)
        else:
            # One CE-only backward pass gives G_t; residual-SVD init of A/B; hooks on.
            self.model.begin_task(task_id, train_loader, self.device)
            self.log.info("[task %s] begin_task xong (gradient-SVD init)", task_id)

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
        running: dict[str, float] = {}
        epoch = first_epoch
        try:
            for epoch in range(first_epoch, total_epochs):
                running = {}
                phase_scale = KeepLoRATrainer.phase_scale(
                    epoch,
                    total_epochs,
                    start=self.align_config.phase_stability_start,
                    minimum=self.align_config.phase_stability_min,
                    maximum=self.align_config.phase_stability_max,
                )
                epoch_started = time.perf_counter()
                batch_count = 0
                with tqdm(train_loader, desc=f"task {task_id} epoch {epoch}", leave=False) as progress:
                    for images, labels in progress:
                        stats = trainer.train_batch(
                            images.to(self.device, non_blocking=True),
                            labels.to(self.device),
                            phase_scale=phase_scale,
                        )
                        self.model.update_routing_statistics(task_id)  # online PFD means
                        for key, value in stats.items():
                            running[key] = running.get(key, 0.0) + value / len(train_loader)
                        batch_count += 1
                        progress.set_postfix(
                            ce=f"{stats.get('loss/ce', float('nan')):.4f}",
                            model=f"{stats.get('loss/model', float('nan')):.4f}",
                        )
                epoch_seconds = time.perf_counter() - epoch_started
                running["phase_scale"] = phase_scale
                running["epoch_time_s"] = epoch_seconds
                running["samples_per_sec"] = (
                    batch_count * int(self.train_cfg.batch_size) / epoch_seconds if epoch_seconds > 0 else 0.0
                )
                for key, value in running.items():
                    self.writer.add_scalar(f"{key}/task{task_id}", value, epoch)
                self._append_epoch_log(task_id, epoch, running)
                self.log.info(
                    "[task %s epoch %s] %s (%.1fs, %.1f samp/s)",
                    task_id,
                    epoch,
                    " ".join(f"{k}={v:.4f}" for k, v in running.items()),
                    epoch_seconds,
                    running["samples_per_sec"],
                )
                next_epoch = epoch + 1
                if (
                    self.checkpoint_every_epochs > 0
                    and next_epoch % self.checkpoint_every_epochs == 0
                    and next_epoch < total_epochs
                ):
                    self._save_live_checkpoint(
                        task_id, next_epoch, model_optimizer, alignment_optimizer, next_epoch=next_epoch
                    )
            self._last_task_train_stats = {k: round(float(v), 6) for k, v in running.items()}
        except KeyboardInterrupt:
            self.log.warning(
                "Interrupt tai task %s epoch %s; dang luu checkpoint de resume...", task_id, epoch
            )
            try:
                self._save_live_checkpoint(
                    task_id, epoch, model_optimizer, alignment_optimizer, next_epoch=epoch
                )
            except Exception as save_error:
                self.log.error("Khong the luu checkpoint sau interrupt: %s", save_error)
            raise
        except Exception as error:
            # Any runtime crash (OOM, shape bug, ...) still preserves progress.
            self.log.exception(
                "%s tai task %s epoch %s; dang luu checkpoint de resume...",
                type(error).__name__,
                task_id,
                epoch,
            )
            try:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                self._save_live_checkpoint(
                    task_id, epoch, model_optimizer, alignment_optimizer, next_epoch=max(epoch, 0)
                )
            except Exception as save_error:
                self.log.error("Khong the luu checkpoint sau loi: %s", save_error)
            raise

        # Consolidation: transport old class stats through A, then fit current-task stats.
        self.model.eval()
        features, labels = self._collect_features(train_loader)
        if task_id > 0:
            self.classifier.transport(self.bicycle.old_to_new)  # mu'=A(mu), Sigma'=A(Sigma)A^T
        self.classifier.fit_task(features, labels)
        del features, labels
        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.empty_cache()
        self.model.end_task(task_id)
        self.old_model = self.model.snapshot()  # immutable teacher for the next task

    @torch.inference_mode()
    def _collect_features(self, loader) -> tuple[torch.Tensor, torch.Tensor]:
        """Pooled backbone features for every sample; caller controls train/eval mode."""
        features, labels = [], []
        for images, targets in loader:
            _, batch_features = self.model(images.to(self.device, non_blocking=True))
            features.append(batch_features.cpu())
            labels.append(targets)
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)

    @torch.inference_mode()
    def _evaluate_upto(self, finished_task: int) -> list[float]:
        """CIL accuracy on all seen tasks with the Gaussian classifier (no task-id hint)."""
        self.model.eval()
        row: list[float] = []
        for spec in self.protocol.tasks[: finished_task + 1]:
            _, test_loader = self.data_manager.task_loaders(spec)
            features, labels = self._collect_features(test_loader)
            predictions = self.classifier.predict(features.to(self.device)).cpu()
            row.append(float((predictions == labels).float().mean()))
            self._log_routing_probe(spec.task_id, test_loader)
            del features, labels, predictions
        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.empty_cache()
        self.accuracy_matrix.append(row)
        return row

    @torch.inference_mode()
    def _log_routing_probe(self, eval_task_id: int, test_loader) -> None:

        """Report the mean router weight on each adapter for one probe batch.

        The routed multi-adapter keeps frozen per-task factors and selects them by
        PFD distribution means. If old-task probes are routed away from their own
        adapter after later tasks, that is the measurable cause of the catastrophic
        forgetting seen in the smoke run (e.g. task 0 dropping below the random
        baseline right after task 1). The probe is one extra eval forward per task,
        so the overhead is negligible.
        """
        images, _ = next(iter(test_loader))
        report = self.model.routing_probe(images.to(self.device, non_blocking=True))
        parts = []
        for layer_name in sorted(report):
            weights = report[layer_name]
            own = weights.get(str(eval_task_id), 0.0)
            others = [value for key, value in weights.items() if key != str(eval_task_id)]
            max_other = max(others) if others else 0.0
            parts.append(f"{layer_name}: own={own:.3f} max_other={max_other:.3f}")
            self.writer.add_scalar(f"routing/own_weight/{layer_name}", own, eval_task_id)
        self.log.info("[routing] eval task %s -> %s", eval_task_id, " | ".join(parts))

    @staticmethod
    def _to_cpu_state(obj: object) -> object:
        """Recursively move every tensor inside a nested structure to CPU.

        Used before serialisation so saving a checkpoint never needs extra GPU
        memory (critical when the save is triggered right after a CUDA OOM).
        """
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {key: DirectionOneExperiment._to_cpu_state(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(DirectionOneExperiment._to_cpu_state(item) for item in obj)
        return obj

    @staticmethod
    def _to_plain_config(obj):
        if isinstance(obj, dict):
            return {k: DirectionOneExperiment._to_plain_config(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [DirectionOneExperiment._to_plain_config(v) for v in obj]
        return obj

    def _base_payload(self) -> dict:
        """Common checkpoint content; only compact statistics, never raw samples.

        All tensors are moved to CPU (``model`` / ``bicycle`` state dicts) before
        serialisation; ``memory`` and ``classifier`` are already CPU-only.
        """
        if OmegaConf is not None and hasattr(OmegaConf, "to_container"):
            config_payload = OmegaConf.to_container(self.cfg, resolve=True)
        else:
            config_payload = self._to_plain_config(self.cfg)
        return {
            "config": config_payload,
            "model": self._to_cpu_state(self.model.state_dict()),
            "memory": self.model.memory_state_dict(),
            "bicycle": self._to_cpu_state(self.bicycle.state_dict()),
            "classifier": self.classifier.export_state(),
            "class_order": self.class_order,
            "accuracy_matrix": self.accuracy_matrix,
            "checkpoint_version": CHECKPOINT_VERSION,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
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
            # ``set_rng_state_all`` expects CPU ByteTensors; new checkpoints store
            # CPU RNG states, legacy files may hold CUDA ones — normalise to CPU.
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])

    def _atomic_save(self, payload: dict, target: Path) -> None:
        """Write via a temp file then rename, so a crash cannot corrupt the checkpoint."""
        tmp = target.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(target)

    def _save_boundary_checkpoint(self, finished_task: int) -> None:
        """Persist task completion checkpoint task_<t>.pt and rolling boundary."""
        payload = self._base_payload()
        payload["progress"] = {"status": "task_done", "task_id": finished_task}

        # 1. Save canonical task-by-task checkpoint: task_XX.pt
        self.checkpoint_manager.save_task_checkpoint(finished_task, payload)
        self.log.info("Da luu task_%02d.pt", finished_task)

        # 2. Save rolling boundary & last checkpoint for compatibility
        self._atomic_save(payload, self.boundary_path)
        self._atomic_save(payload, self.last_path)

    def _save_live_checkpoint(
        self,
        task_id: int,
        completed_epochs: int,
        model_optimizer,
        alignment_optimizer,
        next_epoch: int | None = None,
    ) -> None:
        """Rolling mid-task snapshot (model + optimizers + RNG) for exact resume.

        Everything is converted to CPU before serialisation so the save succeeds
        even under heavy GPU memory pressure (e.g. called right after an OOM).
        """
        payload = self._base_payload()
        payload["progress"] = {
            "status": "training",
            "task_id": task_id,
            "next_epoch": int(next_epoch if next_epoch is not None else completed_epochs),
        }
        payload["model_optimizer"] = self._to_cpu_state(model_optimizer.state_dict())
        payload["alignment_optimizer"] = self._to_cpu_state(alignment_optimizer.state_dict())
        payload["rng"] = self._to_cpu_state(self._rng_state())
        self._atomic_save(payload, self.live_path)

    def _apply_state(self, payload: dict, with_snapshot: bool, with_optimizers: bool = False) -> None:
        """Restore one checkpoint payload into the live experiment objects."""
        if [int(c) for c in payload["class_order"]] != [int(c) for c in self.class_order]:
            raise ValueError("Checkpoint class order khong khop protocol hien tai; khong the resume.")
        self.model.trim_structure(payload["model"])
        self.model.restore_structure(payload["model"])
        self.model.load_state_dict(payload["model"])
        self.model.load_memory_state(payload.get("memory", {}))
        # Resume recreates dynamic modules (adapters / PFD means / head) AFTER
        # the model was already moved to the device, so push everything back.
        self.model.to(self.device)
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

        # 1. Check if an interrupted mid-task live checkpoint exists
        if self.live_path.exists():
            # Mid-task resume needs the previous boundary as the frozen teacher.
            if self.boundary_path.exists():
                boundary = torch.load(self.boundary_path, map_location="cpu", weights_only=False)
                self._validate_payload(boundary)
                self._apply_state(boundary, with_snapshot=True)
            payload = torch.load(self.live_path, map_location="cpu", weights_only=False)
            self._validate_payload(payload)
            self._apply_state(payload, with_snapshot=False, with_optimizers=True)
            progress = payload["progress"]
            self.log.info(
                "Resume tu checkpoint_live.pt (task %s, epoch %s)",
                progress["task_id"],
                progress["next_epoch"],
            )
            return int(progress["task_id"]), int(progress["next_epoch"]), True

        # 2. Check task-by-task checkpoints (task_XX.pt or checkpoint_task_XX.pt or checkpoint_boundary.pt)
        next_task_id, latest_cp = self.checkpoint_manager.find_latest_checkpoint()
        if latest_cp is not None and latest_cp.exists():
            payload = torch.load(latest_cp, map_location="cpu", weights_only=False)
            self._validate_payload(payload)
            self._apply_state(payload, with_snapshot=True)
            finished_task_id = int(payload.get("progress", {}).get("task_id", next_task_id - 1))
            self.log.info(
                "Resume tu %s (task %s da hoan thanh -> bat dau task %s)",
                latest_cp.name,
                finished_task_id,
                finished_task_id + 1,
            )
            return finished_task_id + 1, 0, False

        return 0, 0, False


    def _validate_payload(self, payload: dict) -> None:
        """Lightweight sanity checks before applying a checkpoint payload."""
        version = int(payload.get("checkpoint_version", 1))
        if version > CHECKPOINT_VERSION:
            raise ValueError(
                f"Checkpoint version {version} moi hon code (version {CHECKPOINT_VERSION}); "
                "vui long nang cap code hoac xoa checkpoint cu."
            )
        if "checkpoint_version" in payload and version < CHECKPOINT_VERSION:
            self.log.warning(
                "Checkpoint version %s cu hon code (version %s); van tiep tuc neu schema tuong thich.",
                version,
                CHECKPOINT_VERSION,
            )
        required = ("model", "memory", "bicycle", "classifier", "class_order", "progress")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Checkpoint thieu key {missing}; khong the resume.")

    def _log_history(self, finished_task: int, row: list[float]) -> None:
        """Append one forgetting-tracking record per finished task (survives crashes)."""
        summary_so_far = summarize_accuracy_matrix(self.accuracy_matrix)
        record = {
            "task_id": finished_task,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "task_duration_seconds": round(self._task_durations[-1], 2) if self._task_durations else None,
            "train": self._last_task_train_stats,
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
        "distribution/distance_per_dim",
        "distribution/lambda_adaptive",
        "phase_scale",
        "samples_per_sec",
        "epoch_time_s",
    )

    def _append_epoch_log(self, task_id: int, epoch: int, running: dict[str, float]) -> None:
        """Append per-epoch losses to ``train_log.csv`` (append-safe across resumes)."""
        new_file = not self.train_log_path.exists()
        if not new_file and self._epoch_log_header_mismatched():
            self.log.warning(
                "train_log.csv co header cu hon code; cac cot moi "
                "(phase_scale / samples_per_sec / epoch_time_s) se la NaN trong cac dong cu."
            )
        with open(self.train_log_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(["task_id", "epoch", *self._EPOCH_LOG_KEYS])
            values = [round(running.get(key, float("nan")), 6) for key in self._EPOCH_LOG_KEYS]
            writer.writerow([task_id, epoch, *values])

    def _epoch_log_header_mismatched(self) -> bool:
        """True when an existing ``train_log.csv`` has a different column set."""
        try:
            with open(self.train_log_path, newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle))
            return header != ["task_id", "epoch", *self._EPOCH_LOG_KEYS]
        except (StopIteration, OSError):
            return False

    def _write_running_metrics(self, summary: dict[str, float] | None = None) -> None:
        """Rewrite metrics.json + accuracy CSV + summary_report.txt after every finished task."""
        if summary is None:
            summary = summarize_accuracy_matrix(self.accuracy_matrix)
        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "accuracy_matrix": self.accuracy_matrix}, handle, indent=2)
        with open(self.output_dir / "accuracy_matrix.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(self.accuracy_matrix)
        self._write_summary_report(summary)

    def _write_summary_report(self, summary: dict[str, float]) -> None:
        """Format an easy-to-read text summary report with the full accuracy matrix."""
        report_path = self.output_dir / "summary_report.txt"
        experiment_name = str(self.cfg.experiment.name)
        seed = str(self.cfg.experiment.seed)

        num_tasks = len(self.accuracy_matrix)

        lines = [
            "=" * 82,
            "                   BICYC MULTI-ADAPTER EXPERIMENT REPORT",
            "=" * 82,
            f"Experiment  : {experiment_name}",
            f"Seed        : {seed}",
            f"Device      : {self.device}",
            f"Output Dir  : {self.output_dir}",
            f"Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "-" * 82,
            "CIL CORE METRICS SUMMARY:",
            f"  * Final Task Average Accuracy (A_B)       : {summary.get('last_average', 0.0) * 100:.2f}% ({summary.get('last_average', 0.0):.4f})",
            f"  * Cumulative Incremental Average (A_bar)  : {summary.get('incremental_average', 0.0) * 100:.2f}% ({summary.get('incremental_average', 0.0):.4f})",
            f"  * Catastrophic Forgetting (F)             : {summary.get('forgetting', 0.0):.4f}",
            "-" * 82,
            "ACCURACY MATRIX (%):",
            "(Row i = accuracy across seen tasks after training task i)",
            "-" * 82,
        ]

        if num_tasks > 0:
            header_cols = " ".join(f"T{j:<5}" for j in range(num_tasks))
            header = f"Task | {header_cols} | Last_Avg"
            lines.append(header)
            lines.append("-" * len(header))
            for i in range(num_tasks):
                row = self.accuracy_matrix[i]
                row_vals = []
                for j in range(num_tasks):
                    if j < len(row) and not np.isnan(row[j]):
                        row_vals.append(f"{row[j] * 100:5.2f}%")
                    else:
                        row_vals.append("   -  ")
                valid_accs = [row[j] for j in range(min(i + 1, len(row))) if not np.isnan(row[j])]
                row_avg = float(np.mean(valid_accs) * 100) if valid_accs else 0.0
                lines.append(f"T{i:02d}  | " + " ".join(row_vals) + f" | {row_avg:5.2f}%")

        lines.append("=" * 82)
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def _write_run_meta(self) -> None:
        """Write ``run_meta.json`` (env + git + resolved config) + ``config_resolved.yaml``."""
        # Repository root: <repo>/src/bicyc_multiadapter/engine/task_loop.py -> parents[3].
        repo_root = Path(__file__).resolve().parents[3]
        meta = collect_environment(repo_root)
        if OmegaConf is not None and hasattr(OmegaConf, "to_container"):
            meta["config"] = OmegaConf.to_container(self.cfg, resolve=True)
            yaml_content = OmegaConf.to_yaml(self.cfg)
        else:
            import yaml
            meta["config"] = self._to_plain_config(self.cfg)
            yaml_content = yaml.safe_dump(meta["config"])
        meta["checkpoint_version"] = CHECKPOINT_VERSION
        with open(self.run_meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, default=str)
        with open(self.output_dir / "config_resolved.yaml", "w", encoding="utf-8") as handle:
            handle.write(yaml_content)

    def _update_run_meta(self, summary: dict[str, float]) -> None:
        """Stamp the final summary into ``run_meta.json`` when the run finishes."""
        if not self.run_meta_path.exists():
            return
        with open(self.run_meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        meta["summary"] = summary
        with open(self.run_meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, default=str)

    def _save_outputs(self, summary: dict[str, float]) -> None:
        """Final checkpoint (bases/factors/classifier only), metrics JSON and accuracy CSV."""
        torch.save(self._base_payload(), self.output_dir / "checkpoint_last.pt")
        self._write_running_metrics(summary)
        self.writer.add_hparams(
            {"experiment": str(self.cfg.experiment.name), "seed": str(self.cfg.experiment.seed)},
            {key: float(value) for key, value in summary.items()},
        )
        self.writer.close()
        self._update_run_meta(summary)

    @torch.no_grad()
    def evaluate_final_from_checkpoint(self, payload: dict) -> dict[str, float]:
        """Reload states from a checkpoint payload and re-score the final model once."""
        self.model.restore_structure(payload["model"])
        self.model.load_state_dict(payload["model"])
        self.model.load_memory_state(payload.get("memory", {}))
        self.model.to(self.device)
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
