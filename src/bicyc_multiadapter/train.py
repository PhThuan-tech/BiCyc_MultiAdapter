"""Hydra training entry point for Direction 1.

Usage (from the repository root):
    python -m bicyc_multiadapter.train experiment=keeplora_bicyc
"""

from __future__ import annotations

import os

# Prevent CUDA memory fragmentation on 15 GB GPUs (Kaggle T4 / P100)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Imported lazily so --help stays fast and heavy deps load after Hydra resolves.
    from bicyc_multiadapter.engine.task_loop import DirectionOneExperiment

    experiment = DirectionOneExperiment(cfg)
    summary = experiment.run()
    experiment.log.info("Run summary: %s", summary)


if __name__ == "__main__":
    main()
