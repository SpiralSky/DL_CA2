"""
Generate random sample images from each trained VAE checkpoint.

For each of the 5 VAE variants (basic_vae, improved_vae, res_vae,
augmented_vae, bcvae) this builds the correct (untrained) architecture
via src.models.autoencoders.model_factory, loads its weights from the
matching checkpoint under ./weights, samples 250 images from the latent
prior, decodes them, and saves them as PNGs into ./build_images/<model_name>/.

Total output: 5 models x 250 images = 1000 images.

Checkpoint format (as saved by save_checkpoint()):
    {"model": model.state_dict(), "history": history}

bcvae is conditional, so a random class label (0..NUM_CLASSES-1) is drawn
per sample and passed to the decoder alongside the latent vector.
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image

from src.models.autoencoders.BetaConditionalVAE import BetaConditionalVAE
from src.models.autoencoders.model_factory import (
    basic_vae,
    improved_basic_vae,
    residual_vae,
    beta_conditional_vae,
)

CHECKPOINT_DIR = Path("weights")
OUTPUT_DIR = Path("build_images")

IMAGES_PER_MODEL = 250
LATENT_DIM = 128
NUM_CLASSES = 10  # TODO: set to the actual number of classes used for bcvae

# Maps model name -> factory that builds an *untrained* instance with the
# same architecture used at training time.
MODEL_FACTORIES = {
    "basic_vae": lambda: basic_vae(latent_dim=LATENT_DIM),
    "improved_vae": lambda: improved_basic_vae(latent_dim=LATENT_DIM),
    "res_vae": lambda: residual_vae(latent_dim=LATENT_DIM),
    "bc_vae": lambda: beta_conditional_vae(num_classes=NUM_CLASSES, latent_dim=LATENT_DIM),
}

MODEL_NAMES = list(MODEL_FACTORIES.keys())


def load_checkpoint(model: nn.Module, checkpoint_path: Path):
    """
    Loads a model from a checkpoint file.
    Includes model history.
    :param model: Model to load on.
    :param checkpoint_path: Path to checkpoint file.
    :return: history dict stored alongside the model weights.
    """
    checkpoint = torch.load(
        checkpoint_path,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    return checkpoint["history"]


def load_model(name: str, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Construct an untrained model of the right architecture, then load
    its weights from checkpoint_path.
    """
    model = MODEL_FACTORIES[name]()
    model.to(device)
    load_checkpoint(model, checkpoint_path)
    model.eval()
    return model


@torch.no_grad()
def generate_for_model(name: str, device: torch.device) -> None:
    checkpoint_path = CHECKPOINT_DIR / f"{name}.pt"
    if not checkpoint_path.exists():
        print(f"  [skip] checkpoint not found: {checkpoint_path}")
        return

    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Generating {IMAGES_PER_MODEL} images for {name} ===")
    model = load_model(name, checkpoint_path, device)

    batch_size = 50
    generated = 0
    idx = 0
    while generated < IMAGES_PER_MODEL:
        n = min(batch_size, IMAGES_PER_MODEL - generated)

        if isinstance(model, BetaConditionalVAE):
            labels = torch.randint(0, NUM_CLASSES, (n,), device=device)
            imgs = model.sample(n, labels, device=device)
        else:
            z = torch.randn(n, LATENT_DIM, device=device)
            imgs = model.decoder(z)

        imgs = imgs.clamp(0, 1).cpu()
        for i in range(n):
            save_image(imgs[i], out_dir / f"{name}_{idx:04d}.png")
            idx += 1

        generated += n

    print(f"  saved {idx} images to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random VAE samples")
    parser.add_argument(
        "--model",
        choices=[*MODEL_NAMES, "all"],
        default="all",
        help="Which VAE to generate images from. Defaults to all.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = MODEL_NAMES if args.model == "all" else [args.model]
    for name in selected:
        generate_for_model(name, device)

    print("\n=== Generation complete ===")


if __name__ == "__main__":
    main()