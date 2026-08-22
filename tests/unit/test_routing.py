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
