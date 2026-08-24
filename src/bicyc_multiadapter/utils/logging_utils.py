"""Logging helpers for Direction-1 experiments.

- ``setup_logging``: console + file logger with timestamps.
- ``collect_environment``: hardware/software/git metadata for reproducibility.
"""

from __future__ import annotations

import datetime
import logging
import platform
import socket
import subprocess
import sys
from pathlib import Path

import timm
import torch

_LOG_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)s - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(name: str, output_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure a logger that writes to both console and ``output_dir / run.log``.

    Returns the logger (idempotent: clears existing handlers so the caller always
    gets a fresh configuration).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATEFMT))
    logger.addHandler(console)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(file_handler)

    return logger


def collect_environment(package_root: Path) -> dict:
    """Collect hardware / software / git metadata for reproducibility.

    ``package_root`` should point to the repository root (where ``.git`` is).
    Tolerates missing git, missing GPU, etc.
    """
    meta: dict = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda or "cpu",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "timm": timm.__version__,
    }
    if torch.cuda.is_available():
        meta["cuda_device_name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        meta["cuda_device_capability"] = f"{cap[0]}.{cap[1]}"
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory
            meta["cuda_total_memory_gib"] = round(total_mem / 1024**3, 2)
        except Exception:
            pass

    # Git revision (tolerate non-git directories or missing git).
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(package_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        meta["git_commit"] = result.stdout.strip() if result.returncode == 0 else None
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(package_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        meta["git_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except Exception:
        meta["git_commit"] = None
        meta["git_dirty"] = None

    return meta