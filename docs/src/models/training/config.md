Module src.models.training.config
=================================

Classes
-------

`TrainConfig(max_epochs: int = 300, warmup_epochs: int = 20, lr: float = 0.001, recon_loss_type: str = 'mse', beta_target: float = 1.0, free_bits: float = 0.0, grad_clip_norm: float = 1.0, early_stopping_patience: int = 15, early_stopping_min_delta: float = 0.01, scheduler_patience: int = 10, scheduler_factor: float = 0.7)`
:   All tunable training knobs in one place, analogous to the arguments you'd
    pass to Keras' model.compile()/fit() -- pass this into fit() instead of
    scattering hyperparameters through the training loop.

    ### Instance variables

    `beta_target: float`
    :

    `early_stopping_min_delta: float`
    :

    `early_stopping_patience: int`
    :

    `free_bits: float`
    :

    `grad_clip_norm: float`
    :

    `lr: float`
    :

    `max_epochs: int`
    :

    `recon_loss_type: str`
    :

    `scheduler_factor: float`
    :

    `scheduler_patience: int`
    :

    `warmup_epochs: int`
    :