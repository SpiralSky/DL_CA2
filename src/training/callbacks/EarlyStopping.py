from io import BytesIO
from typing import Literal

import torch

from src.training.callbacks.Callback import Callback, CallbackSignal


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
        self.split, self.metric_key = monitor

        self.patience = patience
        self.min_delta = min_delta
        self.start_epoch = start_epoch

        self.best_value = float("inf")
        self.best_epoch: int | None = None
        self._wait = 0

        self._best_buffer: BytesIO | None = None


    def save_best_state(self, model) -> None:
        buffer = BytesIO()

        torch.save(model.state_dict(), buffer)

        buffer.seek(0)
        self._best_buffer = buffer


    def restore_best_state(self, model) -> None:
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
        trainer,
        epoch: int,
        history,
        metrics: dict[str, dict[str, float]],
    ) -> CallbackSignal | None:

        if epoch < self.start_epoch:
            return None

        current = history.values(
            self.metric_key,
            self.split,
        )[-1]

        if current < self.best_value - self.min_delta:
            self.best_value = current
            self.best_epoch = epoch
            self._wait = 0

            self.save_best_state(trainer.model)

            return None

        self._wait += 1

        if self._wait < self.patience:
            return None

        return CallbackSignal(stop_training=True)

    def on_train_end(self, trainer, history):
        self.restore_best_state(trainer.model)