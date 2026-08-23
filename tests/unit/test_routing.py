import torch

from bicyc_multiadapter.models.adapters.keeplora import initialize_lora_from_gradient
from bicyc_multiadapter.models.adapters.routing import RoutedKeepLoRALinear


def test_routed_lora_is_function_preserving_when_added() -> None:
    weight = torch.randn(4, 3)
    layer = RoutedKeepLoRALinear(weight, None, alpha=8)
    layer.add_task(0, initialize_lora_from_gradient(torch.randn(4, 3), None, rank=2))
    inputs = torch.randn(5, 4)
    layer.update_distribution(0, inputs)
    assert torch.allclose(layer(inputs), inputs @ weight, atol=1e-6)


def test_routed_forward_supports_token_activations() -> None:
    """Real ViT linears see [batch, tokens, dim]; routed mixing must keep that shape."""
    torch.manual_seed(0)
    layer = RoutedKeepLoRALinear(torch.randn(8, 8), None, alpha=4)
    layer.add_task(0, initialize_lora_from_gradient(torch.randn(8, 8), None, rank=2))
    tokens = torch.randn(4, 5, 8)
    layer.update_distribution(0, tokens.reshape(-1, 8))
    output = layer(tokens)  # routing path active: means registered + adapter present
    assert output.shape == tokens.shape
    # A second task (mean + adapter) must not break the mixing either.
    layer.add_task(1, initialize_lora_from_gradient(torch.randn(8, 8), None, rank=2))
    layer.update_distribution(1, torch.randn(16, 8))
    assert layer(tokens).shape == tokens.shape
