"""Download CIFAR-100 once into ``--root``; repeated calls are a no-op.

Used by the cloud notebooks so an interrupted/resumed run never pays the
download cost twice (the folder also gets persisted via Drive / Save Version).
"""

from __future__ import annotations

import argparse

from torchvision import datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Directory that will hold cifar-100-python/")
    args = parser.parse_args()
    for split, name in ((True, "train"), (False, "test")):
        dataset = datasets.CIFAR100(args.root, train=split, download=True)
        print(f"{name}: {len(dataset)} samples -> {args.root}")


if __name__ == "__main__":
    main()
