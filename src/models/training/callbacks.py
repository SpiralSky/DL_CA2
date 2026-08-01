import copy


class Callback:
    """Base class -- override the hook you need. Mirrors keras.callbacks.Callback's shape."""

    def on_epoch_end(self, epoch, logs, model):
        """Return True to request that training stop."""
        return False


class EarlyStopping(Callback):
    """
    PyTorch has no built-in equivalent of keras.callbacks.EarlyStopping, so this
    reimplements it: stop once `monitor` hasn't improved by at least `min_delta`
    for `patience` consecutive epochs, and (optionally) restore the best weights
    seen once training ends.
    """

    def __init__(self, monitor="val_loss", patience=15, min_delta=0.01, restore_best_weights=True):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best = float("inf")
        self.best_epoch = None
        self.best_state = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs, model):
        current = logs[self.monitor]
        if current < self.best - self.min_delta:
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.wait += 1
        return self.wait >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)