from pathlib import Path

import torch
from torch import nn

from src.models.util.TrainingHistory import TrainingHistory


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
) -> TrainingHistory:
    checkpoint = torch.load(
        checkpoint_path,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])

    return checkpoint["history"]


def save_checkpoint(
    model: nn.Module,
    history: TrainingHistory,
    checkpoint_path: Path,
) -> None:
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "history": history,
        },
        checkpoint_path,
    )