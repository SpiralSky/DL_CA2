import io
from io import BytesIO
from pathlib import Path

import torch


class Callback:
    """Base class -- override the hook you need. Mirrors keras.callbacks.Callback's shape."""

    def on_epoch_end(self, epoch, logs, model):
        """Return True to request that training stop."""
        return False

class EarlyStopping(Callback):
    """
    Stop training once `monitor` hasn't improved by at least `min_delta`
    for `patience` consecutive epochs. Optionally restore best weights.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 15,
        min_delta: float = 0.01,
    ) -> None:
        self.monitor_stat: str = monitor
        self.patience: int = patience
        self.min_delta: float = min_delta

        self.best_value: float = float("inf")
        self.best_epoch: int | None = None
        self._wait: int = 0
        self._best_buffer: BytesIO | None = None

    def is_better(self, current: float) -> bool:
        """Check if current metric improved over best by at least min_delta."""
        return current < (self.best_value - self.min_delta)

    def save_checkpoint(self, model, save_path: Path | None = None) -> None:
        """Serialize model state dict to an in-memory buffer and optionally to file."""
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        self._best_buffer = buffer

        if save_path is not None:
            torch.save(model.state_dict(), save_path)

    def load_checkpoint(self, model, save_path: Path | None = None) -> None:
        """Restore model weights from the in-memory buffer or file."""
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
            logs: dict,
            model,
    ) -> str | None:
        current = logs[self.monitor_stat]

        if self.is_better(current):
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