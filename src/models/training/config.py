from dataclasses import dataclass


@dataclass
class TrainConfig:
    """
    All tunable training knobs in one place, analogous to the arguments you'd
    pass to Keras' model.compile()/fit() -- pass this into fit() instead of
    scattering hyperparameters through the training loop.
    """
    max_epochs: int = 300
    warmup_epochs: int = 20
    lr: float = 1e-3
    recon_loss_type: str = "mse"

    beta_target: float = 1.0   # final beta after warmup; lower (e.g. 0.5) to fight posterior collapse
    free_bits: float = 0.0     # per-dim KL allowance before penalty applies; 0 = disabled (original behavior)

    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.01
    scheduler_patience: int = 10
    scheduler_factor: float = 0.7