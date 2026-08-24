"""Two-optimizer training step for the proposed KeepLoRA + BiCyc hybrid."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bicyc_multiadapter.models.alignment.bicyc import (
    BidirectionalCycle,
    bicyc_loss,
    robust_anti_collapse_loss,
)
from bicyc_multiadapter.models.alignment.distribution import adaptive_alignment_weight


@dataclass(frozen=True)
class KeepLoRABiCycConfig:
    lambda_bi: float = 8.0
    lambda_cyc: float = 2.0
    lambda_iso: float = 0.1
    lambda_min: float = 0.4
    lambda_max: float = 1.0
    distribution_temperature: float = 1.0
    use_adaptive_gate: bool = True   # False pins lambda_t=1 (fixed-BiCyc ablation)
    channelwise_gate: bool = True    # True uses per-dimension vector gate [feature_dim]
    use_distillation: bool = True    # False => KeepLoRA-only baseline (ablation 1)
    anti_collapse_weight: float = 0.0
    use_amp: bool = False            # mixed precision for both steps (CUDA only)
    amp_dtype: str = "float16"       # "float16" (T4/RTX, needs GradScaler) or "bfloat16"
    phase_stability_start: float = 0.35
    phase_stability_min: float = 0.75
    phase_stability_max: float = 1.0


_AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


class KeepLoRATrainer:
    """Gradient-safe update schedule (see docs/DIRECTION1_SPEC.md).

    Two strictly separated steps per batch:
      1. model step: CE (+ lambda_t*lambda_bi*||D(z_new)-sg(z_old)||^2) -> B/head;
      2. alignment step: full BiCyc on detached features -> A/D maps only.
    The old model is an immutable task-boundary snapshot and may be ``None``
    for the very first task (plain CE fine-tuning).
    """

    def __init__(
        self,
        current_model: nn.Module,
        old_model: nn.Module | None,
        bicycle: BidirectionalCycle,
        model_optimizer: torch.optim.Optimizer,
        alignment_optimizer: torch.optim.Optimizer,
        config: KeepLoRABiCycConfig,
    ) -> None:
        self.current_model = current_model
        self.old_model = None if old_model is None else old_model.eval()
        self.bicycle = bicycle
        self.model_optimizer, self.alignment_optimizer, self.config = model_optimizer, alignment_optimizer, config
        for parameter in self.bicycle.parameters():
            parameter.requires_grad_(True)
        # AMP is opt-in and silently disabled off-CUDA; statistics (gate, PFD, SVD)
        # always run in fp32 via explicit casts in the loss/statistics helpers.
        self.amp_dtype = _AMP_DTYPES.get(config.amp_dtype, torch.float16)
        self.amp_enabled = bool(config.use_amp and torch.cuda.is_available())
        scaler_enabled = self.amp_enabled and self.amp_dtype == torch.float16
        self.scaler_model = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        self.scaler_alignment = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled)

    @staticmethod
    def _set_trainable(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    @staticmethod
    def phase_scale(
        epoch_index: int,
        total_epochs: int,
        start: float = 0.35,
        minimum: float = 0.75,
        maximum: float = 1.0,
    ) -> float:
        """A simple phase schedule: early epochs prioritize plasticity, later epochs stabilize old tasks."""
        if total_epochs <= 1:
            return maximum
        progress = (epoch_index + 1) / total_epochs
        if progress <= start:
            return minimum + (maximum - minimum) * (progress / max(start, 1e-6))
        return maximum

    def train_batch(self, images: Tensor, labels: Tensor, phase_scale: float = 1.0) -> dict[str, float]:
        phase_scale = float(max(0.0, min(phase_scale, 1.0)))
        old_features = None
        if self.old_model is not None:
            with torch.no_grad(), self._autocast():
                _, old_features = self.old_model(images)

        # --- step 1: CE (+ gated backward distillation) updates B factors and head.
        self._set_trainable(self.bicycle, False)
        self.model_optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            logits, new_features = self.current_model(images)
            classification = F.cross_entropy(logits, labels)
            model_loss = classification
            device = images.device
            adaptive = torch.tensor(1.0, device=device)
            distance_raw = torch.tensor(0.0, device=device)
            distance_per_dim = torch.tensor(0.0, device=device)
            backward_term = torch.tensor(0.0, device=device)
            if old_features is not None and self.config.use_distillation:
                pred_old = self.bicycle.new_to_old(new_features)
                diff_sq = (pred_old - old_features.detach()).square()
                backward_term = diff_sq.mean()
                if self.config.use_adaptive_gate:
                    # fp32 statistics: KL over 768 dims can overflow in half precision.
                    # Per-dimension KL avoids saturation and supports fine-grained channel weighting.
                    adaptive, distance_per_dim, distance_raw = adaptive_alignment_weight(
                        old_features.float(),
                        new_features.float(),
                        self.config.lambda_min,
                        self.config.lambda_max,
                        self.config.distribution_temperature,
                        channelwise=self.config.channelwise_gate,
                    )
                    adaptive = adaptive * phase_scale
                    if adaptive.ndim > 0:
                        distill_loss = (diff_sq * adaptive).mean()
                    else:
                        distill_loss = backward_term * adaptive
                else:
                    distill_loss = backward_term
                model_loss = classification + self.config.lambda_bi * distill_loss
                if self.config.anti_collapse_weight > 0:
                    # Cholesky requires fp32 for numerical stability.
                    model_loss = model_loss + self.config.anti_collapse_weight * robust_anti_collapse_loss(
                        new_features.float()
                    )
        self.scaler_model.scale(model_loss).backward()
        self.scaler_model.step(self.model_optimizer)
        self.scaler_model.update()
        if old_features is None:
            return {"loss/ce": float(classification.detach()), "loss/model": float(model_loss.detach())}

        # --- step 2: all feature tensors detached; this step only learns A/D maps.
        self._set_trainable(self.bicycle, True)
        self.alignment_optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            alignment_terms = bicyc_loss(self.bicycle, old_features.detach(), new_features.detach())
            alignment_loss = alignment_terms.total(
                self.config.lambda_bi * phase_scale,
                self.config.lambda_cyc * phase_scale,
                self.config.lambda_iso * phase_scale,
            )
        self.scaler_alignment.scale(alignment_loss).backward()
        self.scaler_alignment.step(self.alignment_optimizer)
        self.scaler_alignment.update()
        return {
            "loss/ce": float(classification.detach()),
            "loss/model": float(model_loss.detach()),
            "loss/alignment": float(alignment_loss.detach()),
            "loss/backward": float(backward_term.detach()),
            "distribution/distance": float(distance_raw),
            "distribution/distance_per_dim": float(distance_per_dim),
            "distribution/lambda_adaptive": float(adaptive.mean().detach() if isinstance(adaptive, Tensor) else adaptive),
        }
