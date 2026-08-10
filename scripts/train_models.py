import argparse
import gc
from pathlib import Path

import torch

from src.training.autoencoders.base_vae import (
    train_base_vae,
    train_improved_base_vae,
    train_res_vae,
    train_augmented_base_vae,
)
from src.training.autoencoders.conditional_vae import train_bcvae


DATA_PATH = Path("/workspace/DL_CA2/data")
CHECKPOINT_DIR = Path("/workspace/DL_CA2/checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAE model")

    trainers = {
        "basic_vae": train_base_vae,
        "improved_vae": train_improved_base_vae,
        "res_vae": train_res_vae,
        "augmented_vae": train_augmented_base_vae,
        "bcvae": train_bcvae,
    }

    parser.add_argument(
        "--model",
        required=True,
        choices=trainers.keys(),
        help="VAE experiment to train",
    )

    parser.add_argument(
        "--override",
        action="store_true",
        help="Ignore existing checkpoint and overwrite after training",
    )

    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = trainers[args.model]
    checkpoint_path = CHECKPOINT_DIR / f"{args.model}.pt"

    print(f"\n=== Training {args.model} ===")

    trainer(
        data_path=DATA_PATH,
        checkpoint_path=checkpoint_path,
        override=args.override,
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()