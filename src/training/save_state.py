from pathlib import Path

import torch
from torch import nn

from src.models.util.TrainingHistory import TrainingHistory


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
) -> TrainingHistory:
    """
    Loads a model from a checkpoint file.
    Includes model history.
    :param model: Model to load on.
    :param checkpoint_path: Path to checkpoint file.
    :return: None
    """
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
    """
    Saves a model to a checkpoint file.
    :param model: Model to save from.
    :param history: Model history.
    :param checkpoint_path: Path to checkpoint file.
    :return: None
    """
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