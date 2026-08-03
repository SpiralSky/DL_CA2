Module src.models.training.trainer
==================================

Functions
---------

`beta_schedule(epoch: int, warmup_epochs: int, beta_target: float) ‑> float`
:   

`fit(model: torch.nn.modules.module.Module, train_loader: torch.utils.data.dataloader.DataLoader, val_loader: torch.utils.data.dataloader.DataLoader, device: torch.device, max_epochs: int, lr: float, grad_clip_norm: float, warmup_epochs: int = 0, beta_target: float = 1.0, recon_loss_type: str = 'mse', free_bits: float = 0.0, scheduler_patience: int = 5, scheduler_factor: float = 0.5, early_stopping_patience: int = 10, early_stopping_min_delta: float = 0.0, *, optimizer: torch.optim.optimizer.Optimizer | None = None, scheduler: torch.optim.lr_scheduler._LRScheduler | None = None, early_stopping: src.models.training.callbacks.EarlyStopping | None = None, beta_schedule_fn: Callable[[int, int, float], float] = <function beta_schedule>, run_epoch_fn: Callable = <function run_epoch>) ‑> list[dict]`
:   

`run_epoch(model: torch.nn.modules.module.Module, loader: torch.utils.data.dataloader.DataLoader, device: torch.device, optimizer: torch.optim.optimizer.Optimizer, beta: float, config: dict, train: bool) ‑> dict[str, float]`
: