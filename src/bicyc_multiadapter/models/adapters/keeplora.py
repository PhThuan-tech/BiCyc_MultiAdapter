"""KeepLoRA primitives (Luo et al., ICLR 2026).

Matrix convention: ``x @ (W + alpha/r * A @ B)``, W:[d_in,d_out],
A:[d_in,r], B:[r,d_out]. Transpose PyTorch Linear weights before using it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class KeepLoRAFactors:
    """Frozen A and initially trainable B selected from the residual gradient."""

    A: Tensor
    B: Tensor
    projected_gradient: Tensor


def _energy_rank(singular_values: Tensor, energy: float) -> int:
    if not 0 < energy <= 1:
        raise ValueError("energy must be in (0, 1].")
    squared = singular_values.square()
    return int(torch.searchsorted(squared.cumsum(0), energy * squared.sum()).item() + 1)


def principal_weight_subspace(weight: Tensor, energy: float) -> Tensor:
    """Return W_p, the weight principal basis in R^[d_in,p]."""
    U, S, _ = torch.linalg.svd(weight, full_matrices=False)
    return U[:, : _energy_rank(S, energy)]


def orthonormal_union(*bases: Tensor | None) -> Tensor | None:
    """Build Q for the protected union; QR avoids double-projection overlap."""
    nonempty = [basis for basis in bases if basis is not None and basis.numel()]
    if not nonempty:
        return None
    return torch.linalg.qr(torch.cat(nonempty, dim=1), mode="reduced").Q


def residual_gradient(gradient: Tensor, protected_basis: Tensor | None) -> Tensor:
    """G_hat=(I-QQ^T)G, the stable form of KeepLoRA residual projection."""
    if protected_basis is None or protected_basis.numel() == 0:
        return gradient
    return gradient - protected_basis @ (protected_basis.T @ gradient)


def update_feature_subspace(
    input_features: Tensor,
    weight_basis: Tensor,
    historical_basis: Tensor | None,
    energy: float,
) -> Tensor:
    """Append dominant residual input-feature directions to KeepLoRA memory.

    ``input_features`` has shape [batch, d_in]. The paper stores directions in
    the layer input space, not raw images or old examples. Returned columns are
    orthonormal and can be checkpointed as compact task statistics.
    """
    if input_features.ndim != 2 or input_features.shape[1] != weight_basis.shape[0]:
        raise ValueError("Expected input features [batch, d_in] matching weight basis.")
    protected = orthonormal_union(weight_basis, historical_basis)
    residual_inputs = residual_gradient(input_features.detach().T, protected)
    U, S, _ = torch.linalg.svd(residual_inputs, full_matrices=False)
    new_basis = U[:, : _energy_rank(S, energy)]
    return orthonormal_union(historical_basis, new_basis)  # type: ignore[return-value]


def initialize_lora_from_gradient(
    gradient: Tensor, protected_basis: Tensor | None, rank: int
) -> KeepLoRAFactors:
    """Residual-gradient SVD initialization from KeepLoRA Eq. (6).

    The base weight must be offset before the first forward pass;
    :class:`FrozenAResidualLoRA` does this automatically.
    """
    if rank <= 0:
        raise ValueError("rank must be positive.")
    projected = residual_gradient(gradient, protected_basis)
    U, S, Vh = torch.linalg.svd(projected, full_matrices=False)
    effective_rank = min(rank, S.numel())
    return KeepLoRAFactors(
        A=U[:, :effective_rank],
        B=S[:effective_rank].unsqueeze(1) * Vh[:effective_rank, :],
        projected_gradient=projected,
    )


class FrozenAResidualLoRA(nn.Module):
    """KeepLoRA factor with function-preserving *delta* parameterization.

    Algebraically this is the paper's W'=W-alpha/r AB_0 offset: the produced
    update is alpha/r A(B-B_0). It keeps the initial function unchanged while
    allowing the same factor to be used either alone (original KeepLoRA merge)
    or inside a routed multi-adapter bank (the proposed hybrid).
    """

    def __init__(self, weight: Tensor, bias: Tensor | None, factors: KeepLoRAFactors, alpha: float) -> None:
        super().__init__()
        if weight.ndim != 2 or factors.A.shape[0] != weight.shape[0]:
            raise ValueError("Expected W:[d_in,d_out] and compatible KeepLoRA A.")
        self.alpha = float(alpha)
        self.rank = factors.A.shape[1]
        self.register_buffer("A", factors.A.detach().clone())
        self.B = nn.Parameter(factors.B.detach().clone())
        self.register_buffer("initial_B", factors.B.detach().clone())
        self.register_buffer("base_weight", weight.detach().clone())
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def forward(self, inputs: Tensor) -> Tensor:
        bias = 0 if self.bias is None else self.bias
        return inputs @ self.base_weight + self.delta(inputs) + bias

    def delta(self, inputs: Tensor) -> Tensor:
        """The task-specific residual update, initially exactly zero."""
        return self.scaling * (inputs @ self.A @ (self.B - self.initial_B))

    @torch.no_grad()
    def merged_weight(self) -> Tensor:
        return self.base_weight + self.scaling * (self.A @ (self.B - self.initial_B))


class KeepLoRAAdapterBank(nn.Module):
    """Compatibility name; core KeepLoRA merges each task's LoRA instead."""

    def __init__(self, feature_dim: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.feature_dim, self.rank, self.alpha = feature_dim, rank, alpha

    def add_task_adapter(self, *_: object) -> None:
        raise RuntimeError("Use FrozenAResidualLoRA per selected linear layer.")
