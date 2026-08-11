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

BUILD_DIR = ROOT / "build"
IMAGE_DIR = ROOT / "images"

DEVICE = torch.device("cpu")


MODEL_CONFIGS = {
    "basic": (
        basic_vae,
        BUILD_DIR / "basic_vae.pth",
        lambda: torch.randn(1, 3, 32, 32),
    ),
    "improved": (
        improved_basic_vae,
        BUILD_DIR / "improved_vae.pth",
        lambda: torch.randn(1, 3, 32, 32),
    ),
    "residual": (
        residual_vae,
        BUILD_DIR / "residual_vae.pth",
        lambda: torch.randn(1, 3, 32, 32),
    ),
    "bcvae": (
        beta_conditional_vae,
        BUILD_DIR / "bc_vae.pth",
        lambda: (
            torch.randn(1, 3, 32, 32),
            torch.tensor([0]),
        ),
    ),
}


# Tune these independently
GRAPH_CONFIG = {
    "vae": {
        "expand_nested": False,
        "depth": 5,
    },
    "encoder": {
        "expand_nested": True,
        "depth": 5,
    },
    "decoder": {
        "expand_nested": True,
        "depth": 5,
    },
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


def generate_graph(
    model: torch.nn.Module,
    example_input,
    name: str,
    *,
    expand_nested: bool,
    depth: int,
):
    print(f"Generating {name}...")

    graph = draw_graph(
        model,
        input_data=example_input,
        expand_nested=expand_nested,
        depth=depth,
        graph_name=name,
        save_graph=True,
        directory=str(IMAGE_DIR),
        filename=name,
    )

    graph.visual_graph.render(
        str(IMAGE_DIR / name),
        format="png",
    )


def get_encoder_input(example_input):
    if isinstance(example_input, tuple):
        return example_input[0]

    return example_input


def get_decoder_input(model_name: str):
    latent = torch.randn(1, 128)

    if model_name == "bcvae":
        return (
            latent,
            torch.tensor([0]),
        )

    return latent


def generate_model_graphs(model_name: str):
    model, example_input = load_model(model_name)

    # Full VAE
    generate_graph(
        model,
        example_input,
        f"{model_name}_vae",
        **GRAPH_CONFIG["vae"],
    )

    # Encoder
    generate_graph(
        model.encoder,
        get_encoder_input(example_input),
        f"{model_name}_encoder",
        **GRAPH_CONFIG["encoder"],
    )

    # Decoder
    generate_graph(
        model.decoder,
        get_decoder_input(model_name),
        f"{model_name}_decoder",
        **GRAPH_CONFIG["decoder"],
    )


def main():
    IMAGE_DIR.mkdir(exist_ok=True)

    for model_name in MODEL_CONFIGS:
        generate_model_graphs(model_name)


if __name__ == "__main__":
    main()