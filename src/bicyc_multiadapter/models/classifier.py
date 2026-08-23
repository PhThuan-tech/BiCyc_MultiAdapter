"""Compact Gaussian classifier for exemplar-free CIL (BiCyc, Sec. 3.4).

One (mean, covariance-summary) pair per class is stored instead of raw
samples. When the encoder moves from f_{t-1} to f_t, previously stored
statistics are transported through the learned affine map A (mu' = A mu,
Sigma' = A Sigma A^T) rather than recomputed, which is impossible without
exemplars.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class ClassGaussian:
    """Streaming statistics; ``scatter`` is the deviation outer-product sum."""

    mean: Tensor  # [d]
    scatter: Tensor  # [d, d] (full) or [d] (diagonal)
    count: int


class GaussianCILClassifier:
    """Training-free CIL head: fit per task, transport old classes via A, score by Bayes rule."""

    def __init__(self, feature_dim: int, covariance_mode: str = "full_shared", shrinkage: float = 1e-4) -> None:
        if covariance_mode not in {"full_shared", "diagonal"}:
            raise ValueError("covariance_mode must be 'full_shared' or 'diagonal'.")
        self.feature_dim = feature_dim
        self.covariance_mode = covariance_mode
        self.shrinkage = shrinkage
        self.gaussians: dict[int, ClassGaussian] = {}

    # ------------------------------------------------------------------ fitting
    def fit_task(self, features: Tensor, labels: Tensor) -> None:
        """Estimate statistics of the current task under the final f_t; old stats untouched."""
        for label in labels.unique().tolist():
            member = features[labels == label]
            centered = member - member.mean(0, keepdim=True)
            if self.covariance_mode == "full_shared":
                scatter = centered.T @ centered
            else:
                scatter = centered.square().sum(0)
            self.gaussians[int(label)] = ClassGaussian(member.mean(0), scatter.detach(), int(member.shape[0]))

    # --------------------------------------------------------------- transport
    @torch.no_grad()
    def transport(self, projector: nn.Module) -> None:
        """Move every stored statistic into the new space via affine A:z_old->z_new.

        The projector is ``y = x W^T + b`` so mu' = mu W^T + b and the full
        covariance transforms as W Sigma W^T (diagonal: Var_i' = sum_j W_ij^2 Var_j).
        """
        # Gaussian statistics are kept on CPU; bring the projector to CPU so the
        # transport arithmetic (and the loaded checkpoint comparison) stays put.
        weight = projector.net.weight.detach().cpu()
        bias = 0 if projector.net.bias is None else projector.net.bias.detach().cpu()
        for gaussian in self.gaussians.values():
            gaussian.mean = (gaussian.mean @ weight.T + bias).detach().cpu()
            if self.covariance_mode == "full_shared":
                gaussian.scatter = (weight @ gaussian.scatter @ weight.T).detach().cpu()
            else:
                gaussian.scatter = (weight.square() @ gaussian.scatter).detach().cpu()

    # ------------------------------------------------------------------ scoring
    def scores(self, features: Tensor) -> Tensor:
        """Gaussian-Bayes log-likelihoods [batch, C]; argmax is the prediction."""
        classes = sorted(self.gaussians)
        if not classes:
            raise RuntimeError("Fit at least one task before scoring.")
        means = torch.stack([self.gaussians[c].mean for c in classes]).to(features.device)
        covariances = self._covariances(classes, features.device)
        dimension = features.shape[1]
        constant = dimension * math.log(2.0 * math.pi)
        eye_shrink = self.shrinkage * torch.eye(dimension, device=features.device)
        columns = []
        for mean, covariance in zip(means, covariances, strict=True):
            cholesky = torch.linalg.cholesky(covariance + eye_shrink)
            solved = torch.linalg.solve_triangular(cholesky, (features - mean).T, upper=False)
            mahalanobis = solved.square().sum(0)
            log_det = 2.0 * torch.log(cholesky.diagonal()).sum()
            columns.append(-0.5 * (constant + log_det + mahalanobis))
        return torch.stack(columns, dim=1)

    def _covariances(self, classes: list[int], device: str) -> list[Tensor]:
        """Per-class covariance; ``full_shared`` pools within-class scatter (LDA-style)."""
        gaussians = [self.gaussians[c] for c in classes]
        if self.covariance_mode == "full_shared":
            total_count = sum(g.count for g in gaussians)
            pooled = sum((g.scatter for g in gaussians), start=torch.zeros(self.feature_dim, self.feature_dim))
            pooled = pooled / max(total_count - len(classes), 1)
            return [pooled.to(device)] * len(classes)
        # Diagonal variances must be materialized as a [d, d] matrix; adding the
        # 1-D vector to the shrinkage eye would broadcast row-wise into a
        # rank-deficient matrix and break the Cholesky factorization.
        return [torch.diag(g.scatter / max(g.count - 1, 1)).to(device) for g in gaussians]

    def predict(self, features: Tensor) -> Tensor:
        """Argmax mapped back to real class IDs (labels may be any permutation subset)."""
        classes = sorted(self.gaussians)
        best = self.scores(features).argmax(dim=1)
        return torch.as_tensor(classes, device=features.device)[best]

    # ------------------------------------------------------------- persistence
    def export_state(self) -> dict:
        """Serializable payload; only compact statistics, never raw samples."""
        return {
            "covariance_mode": self.covariance_mode,
            "shrinkage": self.shrinkage,
            "feature_dim": self.feature_dim,
            "classes": {
                str(label): {"mean": g.mean.cpu(), "scatter": g.scatter.cpu(), "count": g.count}
                for label, g in self.gaussians.items()
            },
        }

    def load_state(self, payload: dict) -> None:
        self.covariance_mode = payload["covariance_mode"]
        self.shrinkage = payload["shrinkage"]
        self.feature_dim = payload["feature_dim"]
        self.gaussians = {
            int(label): ClassGaussian(item["mean"], item["scatter"], int(item["count"]))
            for label, item in payload["classes"].items()
        }
