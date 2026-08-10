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

WEIGHTS_DIR = ROOT / "weights"
IMAGE_DIR = ROOT / "images"

DEVICE = torch.device("cpu")


def load_model(model_name: str):
    if model_name == "basic":
        model = basic_vae(latent_dim=128)
        checkpoint = WEIGHTS_DIR / "basic_vae.pt"

        example_input = torch.randn(1, 3, 32, 32)

    elif model_name == "improved":
        model = improved_basic_vae(latent_dim=128)
        checkpoint = WEIGHTS_DIR / "improved_vae.pt"

        example_input = torch.randn(1, 3, 32, 32)

    elif model_name == "residual":
        model = residual_vae(latent_dim=128)
        checkpoint = WEIGHTS_DIR / "residual_vae.pt"

        example_input = torch.randn(1, 3, 32, 32)

    elif model_name == "bcvae":
        model = beta_conditional_vae(
            num_classes=10,
            latent_dim=128,
        )
        checkpoint = WEIGHTS_DIR / "bc_vae.pt"

        example_input = (
            torch.randn(1, 3, 32, 32),
            torch.tensor([0]),
        )

    else:
        raise ValueError(model_name)

    state = torch.load(
        checkpoint,
        map_location=DEVICE,
    )

    # handles checkpoints that store model state
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)

    model.eval()

    return model, example_input


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