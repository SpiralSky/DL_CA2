import io
from io import BytesIO
from typing import Literal

import torch
from torch import nn

from src.models.util.TrainingHistory import TrainingHistory
from src.training.callbacks.Callback import Callback


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

        :param monitor: Tuple of (split, metric_key) to monitor.
        :param patience: Number of epochs to wait for improvement before stopping.
        :param min_delta: Minimum change to qualify as improvement.
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


    def save_best_state(self, model: nn.Module) -> None:
        """
        Saves current model state internally.

        :param model: Model to save.
        """
        buffer = io.BytesIO()

        torch.save(
            model.state_dict(),
            buffer,
        )

        buffer.seek(0)

        self._best_buffer = buffer


    def restore_best_state(self, model: nn.Module) -> None:
        """
        Restores best model state observed during training.

        :param model: Model to restore.
        """
        if self._best_buffer is None:
            return

        self._best_buffer.seek(0)

        state_dict = torch.load(
            self._best_buffer,
            weights_only=True,
        )

        model.load_state_dict(state_dict)


    def on_epoch_end(
        self,
        epoch: int,
        history: TrainingHistory,
        model: nn.Module,
    ) -> str | None:
        """
        Called when an epoch ends.

        :param epoch: Current epoch.
        :param history: Model history.
        :param model: Model.
        :return: Early stopping message if triggered.
        """

        if epoch < self.start_epoch:
            return None


        current = history.values(
            self.metric_key,
            self.split,
        )[-1]


        if current < (self.best_value - self.min_delta):
            self.best_value = current
            self.best_epoch = epoch
            self._wait = 0

            self.save_best_state(model)

            return None


        self._wait += 1


        if self._wait < self.patience:
            return None


        return (
            f"\nEarly stopping at epoch {epoch} "
            f"(no improvement > {self.min_delta} "
            f"for {self.patience} epochs)"
        )