from pathlib import Path

import torch
from torchview import draw_graph

from src.models.autoencoders.model_factory import (
    basic_vae,
    improved_basic_vae,
    residual_vae,
    beta_conditional_vae,
)


ROOT = Path(__file__).resolve().parents[1]

WEIGHTS_DIR = ROOT / "build"
IMAGE_DIR = ROOT / "images"

DEVICE = torch.device("cpu")


MODEL_CONFIGS = {
    "basic": (
        basic_vae,
        WEIGHTS_DIR / "basic_vae.pth",
        lambda: torch.randn(1, 3, 32, 32),
    ),
    "improved": (
        improved_basic_vae,
        WEIGHTS_DIR / "improved_vae.pth",
        lambda: torch.randn(1, 3, 32, 32),
    ),
    "residual": (
        residual_vae,
        WEIGHTS_DIR / "residual_vae.pth",
        lambda: torch.randn(1, 3, 32, 32),
    ),
    "bcvae": (
        beta_conditional_vae,
        WEIGHTS_DIR / "bc_vae.pth",
        lambda: (
            torch.randn(1, 3, 32, 32),
            torch.tensor([0]),
        ),
    ),
}


def load_model(model_name: str):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_CONFIGS)}"
        )

    constructor, checkpoint, input_factory = MODEL_CONFIGS[model_name]

    if model_name == "bcvae":
        model = constructor(
            num_classes=10,
            latent_dim=128,
        )
    else:
        model = constructor(
            latent_dim=128,
        )

    state = torch.load(
        checkpoint,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)

    model.eval()

    return model, input_factory()


def main():
    IMAGE_DIR.mkdir(exist_ok=True)

    model, example_input = load_model("improved")

    graph = draw_graph(
        model,
        input_data=example_input,
        expand_nested=True,
        graph_name="VAE Architecture",
        save_graph=True,
        directory=str(IMAGE_DIR),
        filename="vae_architecture",
    )

    graph.visual_graph.render(
        str(IMAGE_DIR / "vae_architecture"),
        format="png",
    )


if __name__ == "__main__":
    main()