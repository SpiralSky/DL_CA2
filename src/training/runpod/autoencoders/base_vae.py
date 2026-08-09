import argparse
import gc
from pathlib import Path

import torch

from src.training.autoencoders.base_vae import (
    train_base_vae,
    train_improved_base_vae,
    train_skip_vae,
)

DATA_PATH = Path("/workspace/DL_CA2/data")
CHECKPOINT_DIR = Path("/workspace/DL_CA2/checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAE model")
    parser.add_argument(
        "--model",
        required=True,
        choices=["basic", "improved", "skip"],
        help="VAE model to train",
    )
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    trainers = {
        "basic": (
            train_base_vae,
            "base_vae.pt",
        ),
        "improved": (
            train_improved_base_vae,
            "improved_vae.pt",
        ),
        "skip": (
            train_skip_vae,
            "skip_vae.pt",
        ),
    }

    trainer, checkpoint_name = trainers[args.model]

    print(f"\n=== Training {args.model} VAE ===")

    trainer(
        data_path=DATA_PATH,
        checkpoint_path=CHECKPOINT_DIR / checkpoint_name,
        override=False,
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()