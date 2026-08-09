import argparse
import gc
from pathlib import Path

import torch

from src.training.autoencoders.base_vae import (
    train_base_vae,
    train_improved_base_vae,
    train_res_vae,
)
from src.training.autoencoders.conditional_vae import train_bcvae


DATA_PATH = Path("/workspace/DL_CA2/data")
CHECKPOINT_DIR = Path("/workspace/DL_CA2/checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAE model")

    parser.add_argument(
        "--model",
        required=True,
        choices=["basic", "improved", "res", "bcvae"],
        help="VAE model to train",
    )

    parser.add_argument(
        "--override",
        action="store_true",
        help="Ignore existing checkpoint and overwrite after training",
    )

    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainers = {
        "basic": (
            train_base_vae,
            "base_vae.pt",
        ),
        "improved": (
            train_improved_base_vae,
            "improved_vae.pt",
        ),
        "res": (
            train_res_vae,
            "res_vae.pt",
        ),
        "bcvae": (
            train_bcvae,
            "bc_vae.pt",
        ),
    }

    trainer, checkpoint_name = trainers[args.model]

    print(f"\n=== Training {args.model} VAE ===")

    trainer(
        data_path=DATA_PATH,
        checkpoint_path=CHECKPOINT_DIR / checkpoint_name,
        override=args.override,
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()