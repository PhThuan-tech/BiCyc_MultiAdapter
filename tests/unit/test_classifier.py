import torch

from bicyc_multiadapter.models.alignment.bicyc import _Projector
from bicyc_multiadapter.models.classifier import GaussianCILClassifier


def test_predict_returns_real_class_ids_for_permuted_label_space() -> None:
    """Class IDs come from a shuffled order; argmax positions must map back to IDs."""
    torch.manual_seed(0)
    classifier = GaussianCILClassifier(4, covariance_mode="diagonal")
    class_ids = [33, 57, 12, 90]  # non-contiguous, as produced by a shuffled protocol
    centers = torch.randn(len(class_ids), 4) * 5
    features = torch.cat([center + 0.1 * torch.randn(10, 4) for center in centers])
    labels = torch.cat([torch.full((10,), cid, dtype=torch.long) for cid in class_ids])
    classifier.fit_task(features, labels)
    predictions = classifier.predict(features)
    assert set(predictions.tolist()) <= set(class_ids)
    assert (predictions == labels).float().mean() > 0.9


def test_predict_maps_back_after_transport_to_new_class_ids() -> None:
    torch.manual_seed(1)
    classifier = GaussianCILClassifier(4, covariance_mode="diagonal")
    old_ids = [5, 17]
    centers = torch.randn(2, 4) * 5
    features_old = torch.cat([center + 0.1 * torch.randn(10, 4) for center in centers])
    labels_old = torch.cat([torch.full((10,), cid, dtype=torch.long) for cid in old_ids])
    classifier.fit_task(features_old, labels_old)

    identity = _Projector(4, None)
    with torch.no_grad():
        identity.net.weight.copy_(torch.eye(4))
        identity.net.bias.zero_()
    classifier.transport(identity)  # A = I keeps the space; IDs must survive
    new_centers = torch.randn(2, 4) * 5
    features_new = torch.cat([center + 0.1 * torch.randn(5, 4) for center in new_centers])
    new_ids = [71, 23]
    labels_new = torch.cat([torch.full((5,), cid, dtype=torch.long) for cid in new_ids])
    classifier.fit_task(features_new, labels_new)

    predictions = classifier.predict(torch.cat([features_old, features_new]))
    assert set(predictions.tolist()) <= {5, 17, 71, 23}
    assert (predictions[:20] == labels_old).float().mean() > 0.9