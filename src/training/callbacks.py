import io

import torch


class Callback:
    """Base class -- override the hook you need. Mirrors keras.callbacks.Callback's shape."""

    def on_epoch_end(self, epoch, logs, model):
        """Return True to request that training stop."""
        return False




class Callback:
    """Base class -- override the hook you need."""

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
        restore_best_weights: bool = True,
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_value = float("inf")
        self.best_epoch = None
        self._wait = 0
        self._best_buffer = None

    def _is_better(self, current: float) -> bool:
        """Check if current metric improved over best by at least min_delta."""
        return current < self.best_value - self.min_delta

    def _save_checkpoint(self, model) -> None:
        """Serialize model state dict to an in-memory buffer."""
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        self._best_buffer = buffer

    def _load_checkpoint(self, model) -> None:
        """Restore model weights from the in-memory buffer."""
        if self._best_buffer is None:
            return
        self._best_buffer.seek(0)
        state_dict = torch.load(self._best_buffer, weights_only=True)
        model.load_state_dict(state_dict)

    def on_epoch_end(self, epoch, logs, model):
        current = logs[self.monitor]

        if self._is_better(current):
            self.best_value = current
            self.best_epoch = epoch
            self._wait = 0
            if self.restore_best_weights:
                self._save_checkpoint(model)
        else:
            self._wait += 1

        return self._wait >= self.patience

    def restore(self, model) -> None:
        """Restore the best weights seen during training, if enabled."""
        if not self.restore_best_weights:
            return
        self._load_checkpoint(model)