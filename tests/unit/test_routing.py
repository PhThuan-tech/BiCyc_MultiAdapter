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


def test_route_report_assigns_own_adapter_the_highest_weight() -> None:
    """Diagnostic probe: a batch from task 0 must route mostly to the task-0 adapter.

    This is the measurable signal for the routed-adapter interference suspected in
    the smoke run, where task 0 accuracy collapsed below random after task 1.
    """
    torch.manual_seed(0)
    layer = RoutedKeepLoRALinear(torch.randn(8, 8), None, alpha=4)
    layer.add_task(0, initialize_lora_from_gradient(torch.randn(8, 8), None, rank=2))
    layer.add_task(1, initialize_lora_from_gradient(torch.randn(8, 8), None, rank=2))
    probe = torch.randn(16, 8)
    layer.update_distribution(0, probe)
    layer.update_distribution(1, probe + 1.0)  # separable second-task mean
    report = layer.route_report(probe)
    assert set(report) == {"0", "1"}
    assert 0.0 <= report["0"] <= 1.0 and 0.0 <= report["1"] <= 1.0
    assert torch.allclose(torch.tensor([report["0"], report["1"]]).sum(), torch.tensor(1.0), atol=1e-6)
    # Task-0 probes stay with the task-0 adapter.
    assert report["0"] > report["1"]
