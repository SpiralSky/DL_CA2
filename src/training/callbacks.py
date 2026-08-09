import io
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from src.models.util.TrainingHistory import TrainingHistory


class Callback:
    """
    Base Callback Class.
    """
    def __init__(self, start_epoch: int = 1) -> None:
        """
        Creates a new callback.
        :param start_epoch: Epoch to start callback at. Callback will only run when current epoch is greater than that epoch.
        """
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch: int, history: TrainingHistory, model: nn.Module) -> Any:
        """
        Intended to be called every epoch.
        :param epoch: Current epoch value.
        :param history: Training history of the model.
        :param model: Model.
        :return:
        """
        return


class EarlyStopping(Callback):
    """
    Callback to stop training early when model stops improving.
    """
    def __init__(
        self,
        monitor: tuple[Literal["train", "val", "extra"], str] = ("val", "loss"),
        patience: int = 15,
        min_delta: float = 0.01,
        start_epoch: int = 1,
    ) -> None:
        """
        Creates new EarlyStopping callback.
        :param monitor: Tuple of (split, metric_key) to monitor,
        e.g. ("val", "loss") or ("train", "acc").
        :param patience: Number of epochs to wait for improvement before stopping.
        :param min_delta: Minimum change to qualify as an improvement.
        :param start_epoch: Epoch to start monitoring at.
        """
        super().__init__(start_epoch)

        self.split, self.metric_key = monitor

        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float("inf")
        self.best_epoch: int | None = None
        self._wait = 0
        self._best_buffer: BytesIO | None = None


    def save_checkpoint(self, model: nn.Module, save_path: Path | None = None) -> None:
        """
        Saves model checkpoint internally.
        :param model: Torch model callback is running on.
        :param save_path: Path to save checkpoint to. If specified, writes the checkpoint to the file.
        :return:
        """
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        self._best_buffer = buffer

        if save_path is not None:
            torch.save(model.state_dict(), save_path)

    def load_checkpoint(self, model: nn.Module, save_path: Path | None = None) -> None:
        """
        Loads checkpoint to current model's state_dict.
        :param model: Model to load checkpoint on.
        :param save_path: If specified, uses this path instead of current callback's best buffer to load weights.
        :return:
        """
        if save_path is not None and save_path.exists():
            state_dict = torch.load(save_path, weights_only=True)
            model.load_state_dict(state_dict)
            return

        if self._best_buffer is None:
            return

        self._best_buffer.seek(0)
        state_dict = torch.load(self._best_buffer, weights_only=True)
        model.load_state_dict(state_dict)

    def on_epoch_end(
        self,
        epoch: int,
        history: TrainingHistory,
        model: nn.Module,
    ) -> str | None:
        """
        Called when and epoch ends and this callback should be run.
        :param epoch: Current epoch.
        :param history: Model History.
        :param model: Model.
        :return:
        """
        if epoch < self.start_epoch:
            return None

        # noinspection bad-argument-type
        current = history.values(self.metric_key, self.split)[-1]

        if current < (self.best_value - self.min_delta):
            self.best_value = current
            self.best_epoch = epoch
            self._wait = 0
            self.save_checkpoint(model)
            return None

        self._wait += 1
        if self._wait < self.patience:
            return None

        return (
            f"\nEarly stopping at epoch {epoch} "
            f"(no improvement > {self.min_delta} "
            f"for {self.patience} epochs)"
        )