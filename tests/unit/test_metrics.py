import torch

from bicyc_multiadapter.evaluation.metrics import forgetting, representation_drift


def test_forgetting_is_non_negative() -> None:
    actual = forgetting(torch.tensor([0.8]), torch.tensor([0.6]))
    assert torch.allclose(actual, torch.tensor([0.2]))


def test_drift_reports_both_metrics() -> None:
    result = representation_drift(torch.ones(2, 3), torch.ones(2, 3))
    assert set(result) == {"l2", "cosine"}
