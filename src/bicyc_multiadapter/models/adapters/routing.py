"""Presentative-feature-distribution routing (Cheng et al., ICML 2025).

The router deliberately has no trainable selection parameters. Each task is
represented by an online mean of frozen-base layer features; inference chooses
and mixes the corresponding frozen KeepLoRA factors.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .keeplora import FrozenAResidualLoRA, KeepLoRAFactors


class PresentativeFeatureRouter(nn.Module):
    """Eq. (3)--(6): online means and L2/dot-product softmax routing."""

    def __init__(self, feature_dim: int, similarity: str = "l2", temperature: float = 1.0) -> None:
        super().__init__()
        if similarity not in {"l2", "dot", "cosine"} or temperature <= 0:
            raise ValueError("similarity must be 'l2', 'dot', or 'cosine'; temperature must be positive.")
        self.feature_dim, self.similarity, self.temperature = feature_dim, similarity, temperature
        self.means = nn.ParameterDict()  # requires_grad=False statistical state, checkpointed with module
        self.counts = nn.ParameterDict()

    def update_distribution(
        self, task_id: int, frozen_base_features: Tensor, batch_count: int | None = None
    ) -> None:
        """Online update of D_k^l=E[W^l h^l(x)]; no raw features are retained."""
        key = str(task_id)
        detached = frozen_base_features.detach()
        if detached.ndim > 2:
            detached = detached.reshape(-1, detached.shape[-1])
        if detached.ndim == 2:
            if detached.shape[1] != self.feature_dim:
                raise ValueError("Expected [batch, feature_dim] frozen-base features.")
            count_val = float(detached.shape[0] if batch_count is None else batch_count)
            batch_mean = detached.mean(0)
        elif detached.ndim == 1:
            if detached.shape[0] != self.feature_dim:
                raise ValueError("Expected [feature_dim] frozen-base features.")
            count_val = float(1 if batch_count is None else batch_count)
            batch_mean = detached
        else:
            raise ValueError("Expected 1D or 2D frozen-base features.")

        if key not in self.means:
            self.means[key] = nn.Parameter(batch_mean.clone(), requires_grad=False)
            self.counts[key] = nn.Parameter(torch.tensor(count_val, device=detached.device), requires_grad=False)
            return
        count = self.counts[key].data
        self.means[key].data.copy_((self.means[key].data * count + batch_mean * count_val) / (count + count_val))
        self.counts[key].data.copy_(count + count_val)

    def routing_weights(self, frozen_base_features: Tensor, top_k: int | None = None) -> tuple[list[str], Tensor]:
        """Return task IDs and per-sample softmax weights, optionally sparse Top-K."""
        task_ids = list(self.means.keys())
        if not task_ids:
            raise RuntimeError("Register a presentative distribution before routing.")
        prototypes = torch.stack([self.means[key] for key in task_ids])
        if self.similarity == "l2":
            scores = -torch.cdist(frozen_base_features, prototypes)
        elif self.similarity == "cosine":
            feat_norm = F.normalize(frozen_base_features, p=2, dim=-1, eps=1e-6)
            proto_norm = F.normalize(prototypes, p=2, dim=-1, eps=1e-6)
            scores = feat_norm @ proto_norm.T
        else:
            scores = frozen_base_features @ prototypes.T / math.sqrt(self.feature_dim)
        if top_k is not None and 0 < top_k < len(task_ids):
            values, indices = scores.topk(top_k, dim=1)
            masked = torch.full_like(scores, -torch.inf)
            scores = masked.scatter(1, indices, values)
        return task_ids, torch.softmax(scores / self.temperature, dim=1)

    @torch.no_grad()
    def forget_task(self, task_id: int) -> None:
        """Drop one task's distribution mean/count (used by the merge baseline)."""
        key = str(task_id)
        del self.means[key]
        del self.counts[key]

    @torch.no_grad()
    def selection_report(self, frozen_base_features: Tensor, top_k: int | None = None) -> dict[str, float]:
        """Mean routing weight per task prototype for one probe batch (diagnostic).

        Lets the runner check whether old-task probes are routed to their own frozen
        adapter or leak to later tasks -- the direct failure mode measured in the
        smoke run (task 0 fell below the random baseline right after task 1).
        """
        task_ids, weights = self.routing_weights(frozen_base_features.detach().float(), top_k)
        means = weights.mean(dim=0)
        return {task_id: float(means[index]) for index, task_id in enumerate(task_ids)}


class RoutedKeepLoRALinear(nn.Module):
    """Proposed multi-adapter extension: KeepLoRA factors + PFD dynamic routing.

    It intentionally does not merge adapters after every task. This is the
    research deviation needed to use the ICML-2025 selection mechanism; old
    factors are frozen and all routing uses non-trainable distribution means.
    """

    def __init__(
        self, weight: Tensor, bias: Tensor | None, alpha: float, similarity: str = "l2", temperature: float = 1.0
    ) -> None:
        super().__init__()
        self.register_buffer("base_weight", weight.detach().clone())
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.alpha = alpha
        self.adapters = nn.ModuleDict()
        self.router = PresentativeFeatureRouter(weight.shape[1], similarity, temperature)
        self.top_k: int | None = None  # default dense mixing; set by the model wrapper

    @torch.no_grad()
    def merge_task(self, task_id: int) -> None:
        """Original KeepLoRA behaviour: fold the task delta back into the base weight."""
        adapter = self.adapters.pop(str(task_id))
        self.base_weight.copy_(adapter.merged_weight())
        self.router.forget_task(task_id)

    def add_task(self, task_id: int, factors: KeepLoRAFactors) -> FrozenAResidualLoRA:
        key = str(task_id)
        if key in self.adapters:
            raise ValueError(f"Task {task_id} already exists.")
        factor = FrozenAResidualLoRA(self.base_weight, None, factors, self.alpha)
        self.adapters[key] = factor
        for parameter in factor.parameters():
            parameter.requires_grad_(True)
        return factor

    def freeze_task(self, task_id: int) -> None:
        for parameter in self.adapters[str(task_id)].parameters():
            parameter.requires_grad_(False)

    def update_distribution(self, task_id: int, layer_inputs: Tensor | tuple[Tensor, int]) -> None:
        """Online update; ``.float()`` keeps means fp32 even under AMP forwards.

        Accepts either:
        - tuple `(mean_vector, batch_count)` where `mean_vector` has shape `[1, d_in]`
          (ultra-compact 3KB representation; avoids 3.25GB GPU VRAM allocation).
        - standard `Tensor` with shape `[batch, d_in]` or `[batch, tokens, d_in]`.
        """
        if isinstance(layer_inputs, tuple):
            mean_vec, batch_count = layer_inputs
            detached = mean_vec.detach().float()
            self.router.update_distribution(task_id, detached @ self.base_weight, batch_count=batch_count)
        else:
            detached = layer_inputs.detach().float()
            if detached.ndim > 2:
                batch_count = detached.shape[0] * detached.shape[1]
                mean_vec = detached.mean(dim=(0, 1), keepdim=True)
                self.router.update_distribution(task_id, mean_vec @ self.base_weight, batch_count=batch_count)
            else:
                self.router.update_distribution(task_id, detached @ self.base_weight)

    @torch.no_grad()
    def route_report(self, inputs: Tensor, top_k: int | None = None) -> dict[str, float]:
        """Per-adapter mean routing weight for one probe batch (diagnostic only)."""
        feats = (inputs.detach().float() @ self.base_weight).float()
        rows = feats.reshape(-1, feats.shape[-1])
        return self.router.selection_report(rows, top_k if top_k is not None else self.top_k)

    def forward(self, inputs: Tensor, top_k: int | None = None) -> Tensor:
        base_features = inputs @ self.base_weight
        if not self.adapters or not self.router.means.keys():
            # Nothing routed yet (task-0 init pass or pre-first PFD update).
            return base_features + (0 if self.bias is None else self.bias)
        # Transformer linears emit [batch, tokens, d_out]; routing statistics are
        # row-based, so flatten tokens, route per row, then restore the layout.
        feats = base_features.detach()
        rows = feats.reshape(-1, feats.shape[-1])
        keys, weights = self.router.routing_weights(rows, top_k if top_k is not None else self.top_k)
        output = base_features
        for index, key in enumerate(keys):
            if key not in self.adapters:
                continue  # defensive: a mean without its adapter (should not happen)
            weight_column = weights[:, index]
            if not (weight_column > 0).any():
                continue
            delta = self.adapters[key].delta(inputs)
            if (weight_column == 1.0).all():
                output = output + delta.to(output.dtype)
            else:
                if base_features.dim() == 3:
                    w = weight_column.view(feats.shape[0], feats.shape[1], 1).to(output.dtype)
                else:
                    w = weight_column.unsqueeze(-1).to(output.dtype)
                output = output + w * delta
        return output + (0 if self.bias is None else self.bias)

