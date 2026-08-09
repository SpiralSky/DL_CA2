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
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Training Basic VAE ===")
    train_base_vae(
        data_path=DATA_PATH,
        checkpoint_path=CHECKPOINT_DIR / "base_vae.pt",
        override=False,
    )
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== Training Improved VAE ===")
    train_improved_base_vae(
        data_path=DATA_PATH,
        checkpoint_path=CHECKPOINT_DIR / "improved_vae.pt",
        override=False,
    )
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== Training Skip VAE ===")
    train_skip_vae(
        data_path=DATA_PATH,
        checkpoint_path=CHECKPOINT_DIR / "skip_vae.pt",
        override=False,
    )
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== All training complete ===")


if __name__ == "__main__":
    main()