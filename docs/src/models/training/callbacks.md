Module src.models.training.callbacks
====================================

Classes
-------

`Callback()`
:   Base class -- override the hook you need.

    ### Descendants

    * src.models.training.callbacks.EarlyStopping

    ### Methods

    `on_epoch_end(self, epoch, logs, model)`
    :   Return True to request that training stop.

`EarlyStopping(monitor: str = 'val_loss', patience: int = 15, min_delta: float = 0.01, restore_best_weights: bool = True)`
:   Stop training once `monitor` hasn't improved by at least `min_delta`
    for `patience` consecutive epochs. Optionally restore best weights.

    ### Ancestors (in MRO)

    * src.models.training.callbacks.Callback

    ### Methods

    `restore(self, model) ‑> None`
    :   Restore the best weights seen during training, if enabled.