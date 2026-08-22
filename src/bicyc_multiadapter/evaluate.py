"""Re-evaluate a finished Direction-1 run from its checkpoint.

Usage:
    python -m bicyc_multiadapter.evaluate experiment=keeplora_bicyc seed=2024
"""

from __future__ import annotations

from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    from bicyc_multiadapter.engine.task_loop import DirectionOneExperiment

    checkpoint = Path(cfg.output_dir) / "checkpoint_last.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Run the training first; missing {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    experiment = DirectionOneExperiment(cfg)
    summary = experiment.evaluate_final_from_checkpoint(payload)
    print("Final evaluation:", summary)


if __name__ == "__main__":
    main()
