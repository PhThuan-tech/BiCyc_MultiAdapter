import torch

from bicyc_multiadapter.models.adapters.keeplora import (
    FrozenAResidualLoRA,
    initialize_lora_from_gradient,
    orthonormal_union,
    residual_gradient,
)


def test_residual_gradient_is_orthogonal_to_protected_basis() -> None:
    protected = torch.tensor([[1.0], [0.0], [0.0]])
    projected = residual_gradient(torch.randn(3, 2), protected)
    assert torch.allclose(protected.T @ projected, torch.zeros(1, 2), atol=1e-6)


def test_keep_lora_offset_preserves_initial_function() -> None:
    torch.manual_seed(1)
    weight = torch.randn(4, 3)
    factors = initialize_lora_from_gradient(torch.randn(4, 3), orthonormal_union(), rank=2)
    module = FrozenAResidualLoRA(weight, None, factors, alpha=8)
    inputs = torch.randn(5, 4)
    assert torch.allclose(module(inputs), inputs @ weight, atol=1e-6)
